import os
import re
import random
import datetime
import glob
import numpy as np
import torch
import wandb

from segment_anything import sam_model_registry
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from DataLoader import TrainingDataset, TestingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, \
    setting_prompt_none, save_masks, parse_train_args, postprocess_masks
from metrics import SegMetrics
from tqdm import tqdm
from pseudo import PseudoSchedular, generate_pseudo

torch.set_default_dtype(torch.float32)
max_num_chkpt = 3


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings,
                       decoder_iter=False):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size),
                          mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions


@torch.no_grad()
def eval_model(args, model, test_loader):
    criterion = FocalDiceloss_IoULoss()
    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}
    test_pbar = tqdm(test_loader)
    l = len(test_loader)
    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
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

            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings)
                if iter != args.iter_point - 1:
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

        masks, pad = postprocess_masks(low_res_masks, args.image_size,
                                       original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size,
                       original_size, pad, batched_input.get("boxes", None),
                       points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in
                              test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]

    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    average_loss = np.mean(test_loss)

    return average_loss, test_iter_metrics


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion,
                    pseudo_schedular):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)

    for batch, batched_input in enumerate(train_loader):

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

        pseudos = (batched_input["pseudo"].unsqueeze(1).
                   repeat(1, args.mask_num).reshape(-1))
        pseudo_weights = torch.ones(size=pseudos.shape)
        pseudo_weights[pseudos] = pseudo_schedular.pseudo_weight

        loss = criterion(masks, labels, iou_predictions, pseudo_weights)
        loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch + 1) % 50 == 0:
            print(
                f'Epoch: {epoch + 1}, Batch: {batch + 1}, '
                f'first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}'
            )

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
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings, decoder_iter=True)

            loss = criterion(masks, labels, iou_predictions, pseudo_weights)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks,
                                               batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

            if int(batch + 1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, '
                        f'mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, '
                        f'point {point_num} prompt: {SegMetrics(masks, labels, args.metrics)}')

        if int(batch + 1) % 200 == 0:
            print(
                f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
                                     f"epoch{epoch + 1}_batch{batch + 1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        train_losses.append(loss.item())

        train_loader.set_postfix(train_loss=loss.item())

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i]
                              for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics


def main(args):

    metric_names = ['iou', 'dice', 'precision', 'f1_score', 'recall',
                    'specificity', 'accuracy', 'hausdorff_distance']

    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    args.run_name = f"{args.run_name}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"

    run_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
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
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            resume_epoch = checkpoint["epoch"]

    params = ["seed", "epochs", "batch_size", "image_size", "mask_num", "lr",
              "resume", "model_type", "sam_checkpoint", "boxes_prompt",
              "point_num", "iter_point", "lr_scheduler", "point_list",
              "multimask", "encoder_adapter"]
    config = {p: getattr(args, p) for p in params}
    config["resume_checkpoint"] = resume_chkpt
    wandb.init(project="SAM_Nuclei", name=args.run_name, config=config)

    # todo: random seed not work
    if args.seed is not None:
        random.seed(args.seed)

    train_dataset = train_dataset_gt = TrainingDataset(
        split_paths=args.split_paths,
        point_num=1,
        mask_num=args.mask_num,
        requires_name=False,
        is_pseudo=False
    )

    # pseudo dataset
    pseudo_schedular = None
    if args.activate_unsupervised:
        pseudo_root = os.path.join(run_dir, "pseudo")
        pseudo_data_dir = os.path.join(pseudo_root, "data")
        os.makedirs(pseudo_data_dir, exist_ok=True)

        generate_pseudo(args, model, pseudo_root)

        pseudo_schedular = PseudoSchedular(
            schedular_dir=run_dir,
            current_epoch=0,
            step=args.unsupervised_step if args.unsupervised_step else 1,
            start_epoch=args.unsupervised_start_epoch,
            pseudo_weight_gr=args.unsupervised_weight_gr
        )

        if resume_epoch >= args.unsupervised_start_epoch:
            pseudo_split_path = os.path.join(pseudo_root, "split.json")
            train_dataset_pseudo = TrainingDataset(
                split_paths=pseudo_split_path,
                point_num=1,
                mask_num=args.mask_num,
                requires_name=False,
                is_pseudo=True
            )
            train_dataset = train_dataset_gt + train_dataset_pseudo

    test_dataset = TestingDataset(split_paths=args.split_paths,
                                  requires_name=True,
                                  point_num=args.point_num,
                                  return_ori_mask=True,
                                  prompt_path=args.prompt_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    loggers = get_logger(
        os.path.join(
            args.work_dir, "logs",
            f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"
        )
    )

    best_loss = 1e10
    l = len(train_loader)

    for epoch in range(resume_epoch, args.epochs):
        print(f"\nTrain epoch {epoch}...")

        if pseudo_schedular is not None:
            pseudo_schedular.step()

        model.train()

        train_losses, train_iter_metrics = train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion,
            pseudo_schedular
        )

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]

        average_loss = np.mean(train_losses)

        print("\nEvaluate model...")
        average_test_loss, test_iter_metrics = eval_model(args, model, test_loader)

        wandb.log({"Loss/train": average_loss, "Loss/test": average_test_loss})

        metrics_dict = dict(zip(metric_names, test_iter_metrics))
        wandb.log(metrics_dict)

        lr = scheduler.get_last_lr()[
            0] if args.lr_scheduler is not None else args.lr
        loggers.info(
            f"epoch: {epoch + 1}, lr: {lr}, train-loss: {average_loss:.4f}, "
            f"test-loss: {average_test_loss:.4f}, metrics: {metrics_dict}")

        if average_test_loss < best_loss:
            # clean redundant checkpoints
            chkpts = sorted(
                glob.glob(os.path.join(run_dir, "*.pth")),
                key=lambda p: float(
                    re.search(r"test-loss(\d+\.\d+)", os.path.basename(p)).group(1))
            )
            for chkpt in chkpts[max_num_chkpt:]:
                os.remove(chkpt)

            # save the latest checkpoint
            best_loss = average_test_loss
            save_path = os.path.join(
                run_dir,
                f"epoch{epoch + 1:04d}_test-loss{average_test_loss:.4f}_sam.pth"
            )
            state = {'model': model.float().state_dict(),
                     'optimizer': optimizer,
                     'train-loss': average_loss, 'test-loss': average_test_loss,
                     'epoch': epoch + 1}
            torch.save(state, save_path)

        if args.activate_unsupervised and pseudo_schedular.is_active():
            pseudo_root = os.path.join(run_dir, "pseudo")
            generate_pseudo(args, model, pseudo_root)
            pseudo_split_path = os.path.join(pseudo_root, "split.json")
            train_dataset_pseudo = TrainingDataset(
                split_paths=pseudo_split_path,
                point_num=1,
                mask_num=args.mask_num,
                requires_name=False,
                is_pseudo=True
            )
            train_dataset = train_dataset_gt + train_dataset_pseudo
            train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)


if __name__ == '__main__':
    args = parse_train_args()
    args.encoder_adapter = True
    # args.activate_unsupervised = True
    main(args)
