import os
import re
import json
import random
import datetime
import glob
import numpy as np
import torch
import wandb

from segment_anything import sam_model_registry
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, TestingDataset, TrainingDatasetFolder, \
    TestingDatasetFolder, stack_dict_batched, CombineBatchSampler
from utils import get_logger, generate_point, setting_prompt_none, save_masks, \
    postprocess_masks, to_device, prompt_and_decoder, MaskPredictor
from loss import FocalDiceloss_IoULoss
from arguments import parse_train_args
from metrics import SegMetrics, AggregatedMetrics
from tqdm import tqdm
from pseudo import PseudoSchedular, generate_pseudo_multiple
import torch.multiprocessing as mp


torch.set_default_dtype(torch.float32)
max_num_chkpt = 3
global_step = 0
global_metrics_dict = {}


@torch.no_grad()
def eval_model(args, model, test_loader, output_dataset_metrics: bool = False):
    model.eval()

    dataset_names = []

    criterion = FocalDiceloss_IoULoss()

    test_loss = []
    prompt_dict = {}
    all_eval_metrics = []

    for i, batched_input in enumerate(tqdm(test_loader)):
        # if i > 5:
        #     break
        batched_input = to_device(batched_input, args.device)
        dataset_names.append(batched_input["dataset_name"])
        ori_labels = batched_input["ori_label"]
        batch_original_size = batched_input["original_size"]
        original_size = batch_original_size[0][0], batch_original_size[1][0]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(
                f"{args.work_dir}", args.run_name,
                f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [
                batched_input["point_labels"]]

            for p in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings)
                if p != args.iter_point - 1:
                    batched_input = generate_point(
                        masks, labels, low_res_masks, batched_input,
                        args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels,dim=1)

            points_show = (torch.concat(point_coords, dim=1),
                           torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size,
                       original_size, pad, batched_input.get("boxes", None),
                       points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        seg_metrics = SegMetrics(args.metrics, masks, labels)
        all_eval_metrics.append(seg_metrics.result())

    average_loss = np.mean(test_loss)

    agg_metrics = AggregatedMetrics(args.metrics, all_eval_metrics, dataset_names)
    metrics_overall = agg_metrics.aggregate()
    if output_dataset_metrics:
        metrics_datasets = agg_metrics.aggregate_by_datasets()
    else:
        metrics_datasets = None

    print("\naggregated whole metrics: ", json.dumps(metrics_overall, indent=2))
    if metrics_datasets is not None:
        print("\naggregated datasets metrics: ", json.dumps(metrics_datasets, indent=2))

    return average_loss, metrics_overall, metrics_datasets


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion,
                    pseudo_schedular, test_loader, pseudo_root, gt_total: int,
                    pseudo_total: int = 0):

    global global_metrics_dict
    global global_step

    if pseudo_schedular is not None:
        pbar = tqdm(total=gt_total + int(pseudo_total * pseudo_schedular.sample_rate))
    else:
        pbar = tqdm(total=gt_total)

    train_loader_bar = tqdm(train_loader)
    train_losses = []

    pseudo_weights = None

    dataloader_iter = iter(train_loader)

    while True:
        pbar.update()
        try:
            batched_input = next(dataloader_iter)
        except StopIteration:
            break

        global_step += 1
        # nn += 1
        # if nn > 16:
        #     break
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        labels = batched_input["label"]
        image_embeddings = model.image_encoder(batched_input["image"])

        batch, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(batch):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)

        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        masks, low_res_masks, iou_predictions = prompt_and_decoder(
            args, batched_input, model, image_embeddings, decoder_iter=False)

        if args.activate_unsupervised and args.unsupervised_weight_gr:
            pseudos = (batched_input["pseudo"].unsqueeze(1).
                       repeat(1, args.mask_num).reshape(-1))
            pseudo_weights = torch.ones(size=pseudos.shape, device=args.device)
            pseudo_weights[pseudos] = pseudo_schedular.pseudo_weight

        loss = criterion(masks, labels, iou_predictions, pseudo_weights)
        loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()
        if pseudo_schedular is not None:
            pseudo_schedular.step(update_epoch=False)

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks,
                                       batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for p in range(args.iter_point):
            if p == init_mask_num or p == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings, decoder_iter=True)

            loss = criterion(masks, labels, iou_predictions, pseudo_weights)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if p != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks,
                                               batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

        if int(batch + 1) % 200 == 0:
            print(
                f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
                                     f"epoch{epoch + 1}_batch{batch + 1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, save_path)

        train_losses.append(loss.item())

        train_loader_bar.set_postfix(train_loss=loss.item())

        if global_step % args.log_interval == 0:
            average_test_loss, test_metrics_overall, test_metrics_datasets = \
                eval_model(args, model, test_loader, output_dataset_metrics=True)

            for metric in args.metrics:
                global_metrics_dict[f"Overall/{metric}"] = test_metrics_overall.get(metric, None)
                if test_metrics_datasets:
                    for dataset_name in test_metrics_datasets.keys():
                        global_metrics_dict[f"{dataset_name}/{metric}"] = \
                            test_metrics_datasets[dataset_name].get(metric, None)

            average_train_loss = np.mean(train_losses)
            global_metrics_dict["Loss/train"] = average_train_loss
            global_metrics_dict["Loss/test"] = average_test_loss

            pseudo_schedular.update_metrics(global_metrics_dict)

            if pseudo_schedular.is_active():
                pseudo_split_path, pseudo_info = generate_pseudo_multiple(
                    args, model, pseudo_root, pseudo_schedular.sample_rate
                )
                for key, val in pseudo_info.items():
                    global_metrics_dict[f"Pseudo/{key}"] = val

                train_loader.batch_sampler.set_sample_rate(pseudo_schedular.sample_rate)
                pbar.total = gt_total + int(pseudo_total * pseudo_schedular.sample_rate)

            wandb.log(global_metrics_dict, step=global_step, commit=True)

            global_metrics_dict = {}
            train_losses = []


def main(args):

    global global_metrics_dict
    global global_step

    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    args.run_name = f"{args.run_name}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"

    run_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10], gamma=0.5)
        print('*******Use MultiStepLR')

    resume_chkpt = None
    if args.resume:
        if os.path.isfile(args.resume):
            resume_chkpt = args.resume
        else:   # dir
            chkpts = sorted(
                glob.glob(os.path.join(run_dir, "*.pth")),
                key=lambda p: float(
                    re.search(r"epoch(\d+)", os.path.basename(p)).group(1))
            )
            if chkpts:
                resume_chkpt = chkpts[-1]

    print("resume_chkpt: ", resume_chkpt)

    resume_epoch = 0
    if resume_chkpt:
        with open(resume_chkpt, "rb") as f:
            checkpoint = torch.load(f, map_location=args.device)
            # model.load_state_dict(checkpoint['model']) --- loaded already using sam_model_registry
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            resume_epoch = checkpoint["epoch"]

    params = ["seed", "epochs", "batch_size", "image_size", "mask_num", "lr",
              "resume", "model_type", "checkpoint", "boxes_prompt",
              "point_num", "iter_point", "lr_scheduler", "point_list",
              "multimask", "encoder_adapter"]
    config = {p: getattr(args, p) for p in params}
    config["resume_checkpoint"] = resume_chkpt
    wandb.init(project="SAM_Nuclei", name=args.run_name, config=config)

    # Random seed Setting
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.data_root:
        train_set_gt = TrainingDatasetFolder(
            data_root=args.data_root,
            train_size=1-args.test_size,
            point_num=1,
            mask_num=args.mask_num,
            requires_name=False,
            random_seed=args.seed
        )
        test_set = TestingDatasetFolder(
            data_root=args.data_root,
            test_size=args.test_size,
            requires_name=True,
            point_num=args.point_num,
            return_ori_mask=True,
            prompt_path=args.prompt_path,
            sample_rate=args.test_sample_rate
        )
    elif args.split_paths:
        train_set_gt = TrainingDataset(
            split_paths=args.split_paths,
            point_num=1,
            mask_num=args.mask_num,
            requires_name=False,
            is_pseudo=False
        )
        test_set = TestingDataset(split_paths=args.split_paths,
                                  requires_name=True,
                                  point_num=args.point_num,
                                  return_ori_mask=True,
                                  prompt_path=args.prompt_path,
                                  sample_rate=args.test_sample_rate)
    else:
        raise ValueError(f"No dataset provided!")

    print("\ngt dataset length: ", len(train_set_gt))

    # pseudo dataset
    pseudo_schedular = None
    pseudo_root = None
    train_set_pseudo = None
    if args.activate_unsupervised:
        pseudo_root = os.path.join(run_dir, "pseudo")
        pseudo_data_dir = os.path.join(pseudo_root, "data")
        os.makedirs(pseudo_data_dir, exist_ok=True)

        pseudo_schedular = PseudoSchedular(
            schedular_dir=run_dir,
            focused_metric=args.unsupervised_focused_metric,
            initial_sample_rate=args.unsupervised_initial_sample_rate,
            sample_rate_delta=args.unsupervised_sample_rate_delta,
            metric_delta_threshold=args.unsupervised_metric_delta_threshold,
            current_epoch=0,
            step=args.unsupervised_step if args.unsupervised_step else 1,
            start_epoch=args.unsupervised_start_epoch,
            pseudo_weight=args.unsupervised_weight,
            pseudo_weight_gr=args.unsupervised_weight_gr
        )

        if pseudo_schedular.is_active():
            print(f"\nactivate pseudo: current step {pseudo_schedular.current_epoch}, "
                  f"weight {pseudo_schedular.pseudo_weight}")
            pseudo_split_path, pseudo_info = generate_pseudo_multiple(
                args, model, pseudo_root, pseudo_schedular.sample_rate
            )
            train_set_pseudo = TrainingDataset(
                split_paths=pseudo_split_path,
                point_num=1,
                mask_num=args.mask_num,
                requires_name=False,
                is_pseudo=True
            )
            for key, val in pseudo_info.items():
                global_metrics_dict[f"Pseudo/{key}"] = val

    if train_set_pseudo is None:
        train_loader = DataLoader(train_set_gt, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    else:
        batch_sampler = CombineBatchSampler(
            gt_dataset_len=len(train_set_gt),
            pseudo_dataset_len=len(train_set_pseudo),
            batch_size=args.batch_size,
            sample_rate=pseudo_schedular.sample_rate,
            drop_last=False
        )
        train_loader = DataLoader(train_set_gt + train_set_pseudo,
                                  batch_sampler=batch_sampler,
                                  num_workers=args.num_workers)

    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    loggers = get_logger(
        os.path.join(
            args.work_dir, "logs",
            f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"
        )
    )

    print("\nEvaluate model using initial checkpoint...")
    average_test_loss, test_metrics_overall, test_metrics_datasets = \
        eval_model(args, model, test_loader, output_dataset_metrics=True)

    for metric in args.metrics:
        global_metrics_dict[f"Overall/{metric}"] = test_metrics_overall.get(metric, None)
        if test_metrics_datasets:
            for dataset_name in test_metrics_datasets.keys():
                global_metrics_dict[f"{dataset_name}/{metric}"] = \
                    test_metrics_datasets[dataset_name].get(metric, None)

    global_metrics_dict["Loss/train"] = 0.0
    global_metrics_dict["Loss/test"] = 0.0

    pseudo_schedular.update_metrics(global_metrics_dict)

    # wandb.log(global_metrics_dict, step=global_step, commit=True)
    wandb.log(global_metrics_dict, step=global_step, commit=True)
    global_metrics_dict = {}
    
    for epoch in range(resume_epoch, args.epochs):
        print(f"\nTrain epoch {epoch}...")

        model.train()

        if pseudo_schedular is not None:
            pseudo_schedular.step(update_epoch=True)

        gt_total = len(train_set_gt)
        pseudo_total = len(train_set_pseudo) if train_set_pseudo is not None else 0

        train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion,
            pseudo_schedular, test_loader, pseudo_root, gt_total, pseudo_total
        )

        if args.lr_scheduler is not None:
            scheduler.step()

        if args.activate_unsupervised:
            global_metrics_dict["Pseudo/weight"] = pseudo_schedular.pseudo_weight

        if args.activate_unsupervised:
            if pseudo_schedular.is_active():
                print(f"\nactivate pseudo: current step {pseudo_schedular.current_epoch}, "
                      f"weight {pseudo_schedular.pseudo_weight}")
                pseudo_root = os.path.join(run_dir, "pseudo")
                pseudo_split_path, pseudo_info = generate_pseudo_multiple(
                    args, model, pseudo_root, pseudo_schedular.sample_rate
                )
                train_set_pseudo = TrainingDataset(
                    split_paths=pseudo_split_path,
                    point_num=1,
                    mask_num=args.mask_num,
                    requires_name=False,
                    is_pseudo=True
                )

                for key, val in pseudo_info.items():
                    global_metrics_dict[f"Pseudo/{key}"] = val
        else:
            train_set_pseudo = None

        if train_set_pseudo is None:
            train_loader = DataLoader(train_set_gt, batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
        else:
            batch_sampler = CombineBatchSampler(
                gt_dataset_len=len(train_set_gt),
                pseudo_dataset_len=len(train_set_pseudo),
                batch_size=args.batch_size,
                sample_rate=pseudo_schedular.sample_rate,
                drop_last=False
            )
            train_loader = DataLoader(train_set_gt + train_set_pseudo,
                                      batch_sampler=batch_sampler,
                                      num_workers=args.num_workers)

        wandb.log(global_metrics_dict, step=global_step, commit=True)
        global_metrics_dict = {}


if __name__ == '__main__':
    mp.set_start_method('spawn')

    args = parse_train_args()

    args.encoder_adapter = True
    # # args.split_paths = ["/Users/zhaojq/Datasets/SAM_nuclei_preprocessed/ALL2/split.json"]

    # args.data_root = "/Users/zhaojq/Datasets/ALL_Multi"
    # args.test_size = 0.1
    # args.checkpoint = "/Users/zhaojq/PycharmProjects/NucleiSAM/pretrain_model/sam_vit_b_01ec64.pth"
    # args.activate_unsupervised = True
    # args.unsupervised_dir = "/Users/zhaojq/Datasets/ALL_Multi/CoNIC/data"
    # args.log_interval = 20
    # args.unsupervised_initial_sample_rate = 0.1
    # # args.unsupervised_weight_gr = 0.1
    # args.batch_size = 4
    # args.num_workers = 1
    # args.points_per_batch = 32
    # args.unsupervised_start_epoch = 1
    # args.unsupervised_sample_rate_delta = 0.1
    # args.unsupervised_metric_delta_threshold = 0.01
    # args.stability_score_thresh = 0.90
    # args.pred_iou_thresh = 0.80
    # args.unsupervised_focused_metric = "Overall/dice"

    main(args)
