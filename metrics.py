import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


metrics_need_pred_mask = ["aji", "dq", "sq", "pq"]


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()


def accuracy(pr, gt, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    correct = (pr_ == gt_).float()
    return torch.mean(correct).cpu().numpy()


def precision(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    tp = torch.sum(gt_ * pr_)
    fp = torch.sum(pr_) - tp
    return (tp + eps) / (tp + fp + eps)


def recall(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    tp = torch.sum(gt_ * pr_)
    fn = torch.sum(gt_) - tp
    return (tp + eps) / (tp + fn + eps)


def f1_score(pr, gt, eps=1e-7, threshold=0.5):
    p = precision(pr, gt, eps, threshold)
    r = recall(pr, gt, eps, threshold)
    #print(p,r)
    return (2 * p * r) / (p + r + eps)


def specificity(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    tn = torch.sum((1 - gt_) * (1 - pr_))
    fp = torch.sum(pr_) - torch.sum(gt_ * pr_)
    return ((tn + eps) / (tn + fp + eps)).cpu().numpy()


# Helper function to convert images to tensors
def to_tensor(img):
    return torch.tensor(np.array(img), device=device)

# Helper function to get instances from a mask
def get_instances(mask):
    instances = []
    unique_labels = torch.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue
        instance = (mask == label)
        instances.append(instance)
    return instances

# Average Jaccard Index (AJI)
def aji(pred, gt, epsilon=1e-10):
    pred_instances = get_instances(pred)
    gt_instances = get_instances(gt)

    intersection_sum = 0
    union_sum = 0

    for gt_instance in gt_instances:
        max_iou = 0
        for pred_instance in pred_instances:
            intersection = torch.sum((gt_instance & pred_instance) > 0)
            union = torch.sum((gt_instance | pred_instance) > 0)
            iou = (intersection + epsilon) / (union + epsilon)
            max_iou = max(max_iou, iou)
        intersection_sum += max_iou
        union_sum += 1

    aji_score = (intersection_sum + epsilon) / (union_sum + epsilon)
    return aji_score

# Detection Quality (DQ)
def dq(pred, gt, epsilon=1e-10):
    pred_instances = get_instances(pred)
    gt_instances = get_instances(gt)

    intersection = torch.zeros((len(gt_instances), len(pred_instances)), device=device)
    union = torch.zeros((len(gt_instances), len(pred_instances)), device=device)

    for i, gt_instance in enumerate(gt_instances):
        for j, pred_instance in enumerate(pred_instances):
            intersection[i, j] = torch.sum((gt_instance & pred_instance) > 0)
            union[i, j] = torch.sum((gt_instance | pred_instance) > 0)

    iou = (intersection + epsilon) / (union + epsilon)
    cost_matrix = 1 - iou.cpu().numpy()
    gt_ind, pred_ind = linear_sum_assignment(cost_matrix)

    dq_score = len(gt_ind) / (len(gt_instances) + len(pred_instances) - len(gt_ind))
    return dq_score

# Segmentation Quality (SQ)
def sq(pred, gt, epsilon=1e-10):
    pred_instances = get_instances(pred)
    gt_instances = get_instances(gt)

    intersection = torch.zeros((len(gt_instances), len(pred_instances)), device=device)
    union = torch.zeros((len(gt_instances), len(pred_instances)), device=device)

    for i, gt_instance in enumerate(gt_instances):
        for j, pred_instance in enumerate(pred_instances):
            intersection[i, j] = torch.sum((gt_instance & pred_instance) > 0)
            union[i, j] = torch.sum((gt_instance | pred_instance) > 0)

    iou = (intersection + epsilon) / (union + epsilon)
    cost_matrix = 1 - iou.cpu().numpy()
    gt_ind, pred_ind = linear_sum_assignment(cost_matrix)

    sq_score = iou[gt_ind, pred_ind].mean().item()
    return sq_score

# Panoptic Quality (PQ)
def pq(pred, gt, epsilon=1e-10):
    dq_score = dq(pred, gt, epsilon)
    sq_score = sq(pred, gt, epsilon)
    pq_score = dq_score * sq_score
    return pq_score


def process_metrics(metric_list):
    processed_list = []
    for item in metric_list:
        if isinstance(item, torch.Tensor):
            # Move tensor to CPU, convert to numpy, then to a plain float
            item = item.cpu().item()  # This works for single element tensors
        elif isinstance(item, float):
            # Handle infinite values if necessary, or leave as is
            if item == float('inf'):
                # Option to handle inf, e.g., set to a large number or np.nan
                item = np.nan  # Using NaN to represent infinite values
        # Append processed item to the new list
        processed_list.append(item)
    
    # Convert list to numpy array for further processing
    metric_array = np.array(processed_list)
    return metric_array


def SegMetrics(pred, pred_mask, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'precision':
            metric_list.append(torch.mean(precision(pred, label)))
        elif metric == 'recall':
            metric_list.append(torch.mean(recall(pred, label)))
        elif metric == 'accuracy':
            metric_list.append(np.mean(accuracy(pred, label)))
        elif metric == 'f1_score':
            metric_list.append(torch.mean(f1_score(pred, label)))
        elif metric == 'specificity':
            metric_list.append(np.mean(specificity(pred, label)))
        elif metric == 'aji':
            metric_list.append(aji(pred_mask, label))
        elif metric == 'dq':
            metric_list.append(dq(pred_mask, label))
        elif metric == 'sq':
            metric_list.append(sq(pred_mask, label))
        elif metric == 'pq':
            metric_list.append(pq(pred_mask, label))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric_list = process_metrics(metric_list)
        #print(f"metric_list:{metric_list}")
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric
