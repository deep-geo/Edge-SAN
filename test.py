import os
import json
import random
import torch
import numpy as np
import wandb

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from arguments import parse_test_args
from utils import generate_point, save_masks,  postprocess_masks
from loss import FocalDiceloss_IoULoss
from DataLoader import TestingDatasetFolder
from metrics import SegMetrics, AggregatedMetrics
from segment_anything import sam_model_registry


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


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            cluster_edges=batched_input.get("edges", None)
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
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
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    wandb.init(project="Nuclei_Test", name=args.run_name)

    model = sam_model_registry[args.model_type](args).to(args.device) 

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDatasetFolder(
        data_root=args.data_root,
        requires_name=True,
        point_num=args.point_num
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Test data:', len(test_loader))

    model.eval()

    all_metrics = []
    test_loss = []
    prompt_dict = {}

    for i, batched_input in enumerate(tqdm(test_loader)):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"][0][0], batched_input["original_size"][1][0]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.save_prompt:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"] = None
            batched_input["point_labels"] = None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings
            )
            points_show = None

        else:
            save_path = os.path.join(
                f"{args.work_dir}",
                args.run_name,
                f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt"
            )
            batched_input["boxes"] = None
            point_coords = [batched_input["point_coords"]]
            point_labels = [batched_input["point_labels"]]
     
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings
                )
                if iter != args.iter_point-1:
                    batched_input = generate_point(
                        masks, labels, low_res_masks, batched_input, args.point_num
                    )
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1),
                           torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(
                masks, save_path, img_name, args.image_size, original_size,
                pad, batched_input.get("boxes", None), points_show
            )

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        seg_metrics = SegMetrics(args.metrics, masks, ori_labels)
        all_metrics.append(seg_metrics.result())

    test_metrics = AggregatedMetrics(args.metrics, all_metrics).aggregate()
    average_loss = np.mean(test_loss)

    wandb.log({"test_loss": average_loss})
    wandb.log(test_metrics)

    if args.save_prompt:
        prompt_path = os.path.join(args.work_dir, f'{args.image_size}_prompt.json')
        with open(prompt_path, 'w') as f:
            json.dump(prompt_dict, f, indent=2)

    print(f"\nTest loss: {average_loss:.4f}")
    print("\nTest metrics: ", json.dumps(test_metrics, indent=2))


if __name__ == '__main__':
    args = parse_test_args()

    # args.data_root = "/Users/zhaojq/Datasets/SAM_nuclei_preprocessed/MoNuSeg2020"

    # Random seed Setting
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)
