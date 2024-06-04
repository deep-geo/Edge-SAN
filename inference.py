import os
import torch
import tqdm
import numpy as np
import datetime
import wandb
import random

from arguments import parse_inference_args
from segment_anything import sam_model_registry
from DataLoader import TestingDatasetFolder
from torch.utils.data import DataLoader
from utils import (generate_point, save_masks, postprocess_masks, to_device,
                   prompt_and_decoder, MaskPredictor)
from loss import FocalDiceloss_IoULoss
from metrics import SegMetrics

torch.set_default_dtype(torch.float32)


@torch.no_grad()
def inference(args, model, data_loader):
    criterion = FocalDiceloss_IoULoss()
    model.eval()

    metrics_data = {_: [] for _ in args.metrics}
    pred_masks = None
    if args.predict_masks:
        mask_predictor = MaskPredictor(
            model=model,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch
        )
    else:
        mask_predictor = None

    losses = []
    miss_rate = []
    prompt_dict = {}

    for i, batched_input in enumerate(tqdm.tqdm(data_loader, desc="Inference")):
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
                args.work_dir,
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

        miss_rate.append(
            ((torch.sum(low_res_masks, dim=(2, 3)) == 0).sum() /
            low_res_masks.shape[0]).item()
        )

        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size,
                       original_size, pad, batched_input.get("boxes", None),
                       points_show)

        if args.predict_masks:
            image_paths = batched_input["image_path"]
            pred_masks = mask_predictor.batch_predict(image_paths)
            pred_masks = torch.tensor(np.array(pred_masks, dtype=np.int32)).unsqueeze(1).to(args.device)

        loss = criterion(masks, ori_labels, iou_predictions)
        losses.append(loss.item())

        test_batch_metrics = SegMetrics(masks, pred_masks, ori_labels, args.metrics)

        for j in range(len(args.metrics)):
            metrics_data[args.metrics[j]].append(test_batch_metrics[j])

    average_metrics = {key: np.mean(vals) for key, vals in metrics_data.items()}
    average_loss = np.mean(losses)
    average_miss_rate = np.mean(miss_rate)

    return average_loss, average_metrics, average_miss_rate


def main(args):
    args.run_name = f"{args.run_name}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    wandb.init(project="NucleiSAM_Inference", name=args.run_name)

    model = sam_model_registry[args.model_type](args).to(args.device)
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location=args.device)
            model.load_state_dict(checkpoint['model'], strict=False)

    dataset = TestingDatasetFolder(data_root=args.data_root,
                                   requires_name=True,
                                   point_num=args.point_num,
                                   return_ori_mask=True,
                                   prompt_path=args.prompt_path)
    print("\nDataset length: ", len(dataset))
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    average_loss, metrics, miss_rate = inference(args, model, data_loader)

    print("average_loss: ", average_loss)
    print("metrics: ", metrics)
    print("miss_rate: ", miss_rate)

    results = {"average_loss": average_loss, "miss_rate": miss_rate}
    results.update(metrics)
    wandb.log(results)
    wandb.finish()


if __name__ == '__main__':
    args = parse_inference_args()
    args.encoder_adapter = True

    seed = args.random_seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # args.batch_size = 1
    # args.data_root = "/Users/zhaojq/Datasets/SAM_nuclei_preprocessed/cpm15"
    # args.pred_iou_thresh = 0.8
    # args.stability_score_thresh = 0.9
    # args.sam_checkpoint = "/Users/zhaojq/PycharmProjects/NucleiSAM/pretrain_model/sam_vit_b_01ec64.pth"
    # args.checkpoint = "epoch0077_test-loss0.1181_sam.pth"

    main(args)
