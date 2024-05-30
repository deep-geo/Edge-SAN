import argparse
import datetime

from metrics import metrics_need_pred_mask


def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir", type=str, default="workdir",
                        help="work dir")
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"run-{str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-')}",
        help="run model name"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="number of epochs")
    parser.add_argument("--epochs", type=int, default=100000,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="train batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Dataloader workers")
    parser.add_argument("--image_size", type=int, default=256,
                        help="image_size")
    parser.add_argument("--mask_num", type=int, default=5,
                        help="get mask number")
    parser.add_argument("--predict_masks", action='store_true',
                        help="predict masks or not")

    parser.add_argument("--split_paths", nargs='+', default=[],
                        help="path list of the split json files")
    parser.add_argument(
        "--metrics",
        nargs='+',
        default=['iou', 'dice', 'precision', 'f1_score', 'recall',
                 'specificity', 'accuracy', 'hausdorff_distance'],
        help="metrics"
    )
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88,
                        help="Mask filtering threshold in [0,1]")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95,
                        help="Mask filtering threshold in [0,1]")
    parser.add_argument("--points_per_side", type=int, default=32,
                        help="arg for mask generator")
    parser.add_argument("--points_per_batch", type=float, default=256,
                        help="arg for mask generator")

    parser.add_argument('--device', type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint or run_name to resume")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="sam checkpoint")
    parser.add_argument("--boxes_prompt", action='store_true',
                        help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1,
                        help="point num")
    parser.add_argument("--iter_point", type=int, default=8,
                        help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9],
                        help="point_list")
    parser.add_argument("--multimask", action='store_true',
                        help="ouput multimask")
    parser.add_argument("--encoder_adapter", action='store_true',
                        help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None,
                        help="fix prompt path")
    parser.add_argument("--save_pred", action='store_true',
                        help="save result")

    # unsupervised
    parser.add_argument("--activate_unsupervised", action="store_true",
                        help="activate unsupervised")
    parser.add_argument("--unsupervised_only", action="store_true",
                        help="activate unsupervised")
    parser.add_argument("--unsupervised_dir", type=str,
                        help="dir cointaining unsupervised data")
    parser.add_argument("--unsupervised_start_epoch", type=int, default=0,
                        help="epoch to start generating unsupervised dataset")
    parser.add_argument("--unsupervised_step", type=int, default=None,
                        help="step to update unsupervised dataset")
    parser.add_argument("--unsupervised_weight_gr", type=float,
                        default=0.1)

    # parser.add_argument("--use_amp", type=bool, default=False, help="use amp")

    args = parser.parse_args()
    if args.resume:
        args.sam_checkpoint = None

    if set(metrics_need_pred_mask) & args.metrics:
        args.predict_masks = True

    return args


def parse_inference_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="run")

    # model
    parser.add_argument("--sam_checkpoint", type=str, help="checkpoint")

    parser.add_argument("--model_type", type=str, default="vit_b",
                        help="sam model_type")
    parser.add_argument('--device', type=str, default='cpu',
                        help="cuda or cpu")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--predict_masks", action='store_true',
                        help="predict masks or not")

    # data loader
    parser.add_argument("--data_root", type=str, default="",
                        help="Directory of the inference dataset.")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=8,
                        help="Dataloader workers")

    # metrics
    parser.add_argument(
        "--metrics",
        nargs='+',
        default=['iou', 'dice', 'precision', 'f1_score', 'recall',
                 'specificity', 'accuracy'],
        help="metrics"
    )
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88,
                        help="Mask filtering threshold in [0,1]")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95,
                        help="Mask filtering threshold in [0,1]")
    parser.add_argument("--points_per_side", type=int, default=32,
                        help="arg for mask generator")
    parser.add_argument("--points_per_batch", type=float, default=256,
                        help="arg for mask generator")

    parser.add_argument("--boxes_prompt", action='store_true',
                        help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1,
                        help="point num")
    parser.add_argument("--iter_point", type=int, default=8,
                        help="point iterations")
    parser.add_argument("--multimask", action='store_true',
                        help="ouput multimask")
    # parser.add_argument("--encoder_adapter", action='store_true',
    #                     help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None,
                        help="fix prompt path")
    parser.add_argument("--save_pred", action='store_true',
                        help="save result")

    args = parser.parse_args()

    if set(metrics_need_pred_mask) & args.metrics:
        args.predict_masks = True

    return args
