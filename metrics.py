import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


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

# def hausdorff_distance(pr, gt, threshold=0.5):
#     pr_, gt_ = _list_tensor(pr, gt)
#     pr_ = _threshold(pr_, threshold=threshold).cpu().numpy()
#     gt_ = _threshold(gt_, threshold=threshold).cpu().numpy()

#     # Calculate Hausdorff distance for each item in the batch
#     distances = [directed_hausdorff(u, v)[0] for u, v in zip(pr_, gt_)]
#     return (torch.mean(distances)).cpu().numpy()


def hausdorff_distance(pr, gt, threshold=0.5):
    # Assuming _list_tensor and _threshold properly prepare and convert tensors
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold).cpu().numpy()
    gt_ = _threshold(gt_, threshold=threshold).cpu().numpy()

    distances = []
    for u, v in zip(pr_, gt_):
        # Convert each binary image to a set of points
        points_u = np.argwhere(u)
        points_v = np.argwhere(v)

        # Check if any of the sets are empty
        if points_u.size == 0 or points_v.size == 0:
            # Can decide on a suitable default distance when one set is empty
            distances.append(float('inf'))  # Consider infinite distance if no points
            continue

        # Calculate directed Hausdorff distances and take the maximum
        dist_uv = directed_hausdorff(points_u, points_v)[0]
        dist_vu = directed_hausdorff(points_v, points_u)[0]
        distances.append(max(dist_uv, dist_vu))

    # Convert list to a tensor to use torch.mean
    distances_tensor = torch.tensor(distances, dtype=torch.float32)
    return torch.mean(distances_tensor).item()  # Returns a single scalar value


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


def SegMetrics(pred, label, metrics):
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
        elif metric == 'hausdorff_distance':
            metric_list.append(np.mean(hausdorff_distance(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric_list = process_metrics(metric_list)
        #print(f"metric_list:{metric_list}")
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric
