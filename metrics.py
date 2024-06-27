import torch
import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = 1e-7

instance_metrics = ["dq", "sq", "pq"]


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


def logits2instances(x, threshold: float = 0.5):
    if type(x) is list:
        x = torch.tensor(np.array(x))
    if x.min() < 0 or x.max() > 1:
        x = torch.nn.Sigmoid()(x)
    return _threshold(x, threshold=threshold)


def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = logits2instances(pr, threshold)
    intersection = torch.sum(gt * pr_, dim=[1, 2, 3])
    union = torch.sum(gt, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = logits2instances(pr, threshold)
    intersection = torch.sum(gt * pr_, dim=[1, 2, 3])
    union = torch.sum(gt, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()


def accuracy(pr, gt, threshold=0.5):
    pr_ = logits2instances(pr, threshold)
    correct = (pr_ == gt).float()
    return torch.mean(correct).cpu().numpy()


def precision(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = logits2instances(pr, threshold)
    tp = torch.sum(gt * pr_)
    fp = torch.sum(pr_) - tp
    return (tp + eps) / (tp + fp + eps)


def recall(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = logits2instances(pr, threshold)
    tp = torch.sum(gt * pr_)
    fn = torch.sum(gt) - tp
    return (tp + eps) / (tp + fn + eps)


def f1_score(pr, gt, eps=1e-7, threshold=0.5):
    p = precision(pr, gt, eps, threshold)
    r = recall(pr, gt, eps, threshold)
    return (2 * p * r) / (p + r + eps)


def specificity(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = logits2instances(pr, threshold)
    tn = torch.sum((1 - gt) * (1 - pr_))
    fp = torch.sum(pr_) - torch.sum(gt * pr_)
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
    pred_instances = logits2instances(pred)
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
def dq(preds, gts, epsilon=1e-10, threshold=0.5):
    preds = logits2instances(preds, threshold)
    pred_instances = get_instances(preds)
    gt_instances = get_instances(gts)

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
def sq(preds, gts, epsilon=1e-10, threshold=0.5):
    preds = logits2instances(preds, threshold)
    pred_instances = get_instances(preds)
    gt_instances = get_instances(gts)

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
def pq(preds, gts, epsilon=1e-10):
    dq_score = dq(preds, gts, epsilon)
    sq_score = sq(preds, gts, epsilon)
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


def SegMetricsFunc(preds, gts, metrics):
    """
    preds & gts shape: B * C * H * W
    gts values: bg 0, fg 1
    """
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(preds, gts)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(preds, gts)))
        elif metric == 'precision':
            metric_list.append(torch.mean(precision(preds, gts)))
        elif metric == 'recall':
            metric_list.append(torch.mean(recall(preds, gts)))
        elif metric == 'accuracy':
            metric_list.append(np.mean(accuracy(preds, gts)))
        elif metric == 'f1_score':
            metric_list.append(torch.mean(f1_score(preds, gts)))
        elif metric == 'specificity':
            metric_list.append(np.mean(specificity(preds, gts)))
        elif metric == 'aji':
            metric_list.append(aji(preds, gts))
        elif metric == 'dq':
            metric_list.append(dq(preds, gts))
        elif metric == 'sq':
            metric_list.append(sq(preds, gts))
        elif metric == 'pq':
            metric_list.append(pq(preds, gts))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if preds is not None:
        metric_list = process_metrics(metric_list)
        # print(f"metric_list:{metric_list}")
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric


class SegMetrics:
    """
    metrics: metrics to calculate
    preds: un-normalized predicted masks which is logits, shape: B * C * H * W
    gts: ground truth masks, shape: B * C * H * W
    threshold: probability to convert sigmoided predicts to masks.
    """

    def __init__(self, metrics: List[str], predicts: torch.Tensor,
                 gts: torch.Tensor, prob_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
        self._metrics = metrics
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.pred_masks = self.get_predicted_masks(predicts)
        self.gts = gts.type(torch.int32)

    def result(self) -> dict:
        result = {
            metric: getattr(self, metric) if hasattr(self, metric) else None
            for metric in self._metrics
        }
        if "aji" in self._metrics:
            if "intersection" not in self._metrics:
                result["intersection"] = self.intersection
            if "union" not in self._metrics:
                result["union"] = self.union

        for metric in ["tp", "fp", "tn", "fn"]:
            result[metric] = getattr(self, metric)

        return result

    def get_predicted_masks(self, predicts: torch.Tensor):
        probs = torch.sigmoid(predicts)
        return (probs > self.prob_threshold).type(torch.int32)

    @property
    def accuracy(self):
        key = "_accuracy"
        if not hasattr(self, key):
            correct = self.pred_masks == self.gts
            value = torch.mean(correct.type(torch.float32), dim=[1, 2, 3])
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def tp(self):
        key = "_tp"
        if not hasattr(self, key):
            value = (self.iou > self.iou_threshold).type(torch.int32)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def fp(self):
        key = "_fp"
        if not hasattr(self, key):
            value = ((self.iou <= self.iou_threshold) & (self.pred_masks.sum(dim=[1, 2, 3]) > 0)).type(torch.int32)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def tn(self):
        key = "_tn"
        if not hasattr(self, key):
            value = ((self.gts.sum(dim=[1, 2, 3]) == 0) & (self.pred_masks.sum(dim=[1, 2, 3]) == 0)).type(torch.int32)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def fn(self):
        key = "_fn"
        if not hasattr(self, key):
            value = ((self.pred_masks.sum(dim=[1, 2, 3]) == 0) & (self.gts.sum(dim=[1, 2, 3]) > 0)).type(torch.int32)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def precision(self):
        key = "_precision"
        if not hasattr(self, key):
            value = self.intersection / (self.pred_masks.sum(dim=[1, 2, 3]) + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def recall(self):
        key = "_recall"
        if not hasattr(self, key):
            # value = self.tp / (self.tp + self.fn + epsilon)
            value = self.intersection / (torch.sum(self.gts, dim=[1, 2, 3]) + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def f1_score(self):
        key = "_f1_score"
        if not hasattr(self, key):
            value = (2 * self.precision * self.recall) / (self.precision + self.recall + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def specificity(self):
        key = "_specificity"
        if not hasattr(self, key):
            value = self.tn / (self.tn + self.fp + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def intersection(self):
        key = "_intersection"
        if not hasattr(self, key):
            value = torch.sum(self.gts & self.pred_masks, dim=[1, 2, 3])
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def union(self):
        key = "_union"
        if not hasattr(self, key):
            value = torch.sum(self.gts | self.pred_masks, dim=[1, 2, 3])
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def iou(self):
        key = "_iou"
        if not hasattr(self, key):
            value = self.intersection / (self.union + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def dice(self):
        key = "_dice"
        if not hasattr(self, key):
            gt_areas = torch.sum(self.gts, dim=[1, 2, 3])
            predict_areas = torch.sum(self.pred_masks, dim=[1, 2, 3])
            value = 2 * self.intersection / (gt_areas + predict_areas + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def dq(self):
        key = "_dq"
        if not hasattr(self, key):
            value = self.tp / (self.tp + (self.fp + self.fn) / 2 + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def sq(self):
        key = "_sq"
        if not hasattr(self, key):
            value = self.iou / (self.tp + epsilon)
            setattr(self, key, value)
        return getattr(self, key)

    @property
    def pq(self):
        key = "_pq"
        if not hasattr(self, key):
            value = self.dq * self.sq
            setattr(self, key, value)
        return getattr(self, key)


class AggregatedMetrics:

    def __init__(self, metrics: List[str], metric_data: List[dict],
                 dataset_names: List[List[str]] = None):
        self.metrics = metrics
        self.metric_data = metric_data
        self.dataset_names = dataset_names

    def _aggregate(self, metrics_data: List[dict]):
        # average
        result = {}
        for metric in self.metrics:
            if metrics_data[0].get(metric, None) is None:
                result[metric] = None
                continue
            if metric != "aji" and metric not in instance_metrics:
                result[metric] = self.average(metric, metrics_data)

        # aji - Aggregated Jaccard Index
        if "aji" in self.metrics:
            result["aji"] = self.sum("intersection", metrics_data) / (self.sum("union", metrics_data) + epsilon)

        # dq
        count_tp = self.sum("tp", metrics_data)
        count_fp = self.sum("fp", metrics_data)
        count_fn = self.sum("fn", metrics_data)
        result["dq"] = count_tp / (count_tp + (count_fp + count_fn) / 2)

        # sq
        iou_arr = torch.cat([_["iou"] for _ in metrics_data]).cpu().numpy()
        tp_arr = torch.cat([_["tp"] for _ in metrics_data]).cpu().numpy()
        result["sq"] = np.sum(iou_arr * tp_arr) / (np.sum(tp_arr) + epsilon)

        # pq
        result["pq"] = result["dq"] * result["sq"]

        result = {key: value.item() for key, value in result.items()}

        return result

    def aggregate(self):
        return self._aggregate(self.metric_data)

    def aggregate_by_datasets(self):
        names = set()
        for name_list in self.dataset_names:
            names = names | set(name_list)
        names = list(names)

        result = {}
        for name in names:
            metrics_data = []
            for metric_dict, name_list in zip(self.metric_data, self.dataset_names):
                check = np.array(name_list) == name
                if not check.any():
                    continue
                metrics_data.append(
                    {
                        key: val[check] if val is not None else None
                        for key, val in metric_dict.items()
                    }
                )
            result[name] = self._aggregate(metrics_data)

        return result

    @staticmethod
    def average(metric: str, metrics_data: List[dict]):
        arr = torch.cat([_[metric] for _ in metrics_data]).cpu().numpy()
        return np.mean(arr)

    @staticmethod
    def sum(metric: str, metrics_data: List[dict]):
        arr = torch.cat([_[metric] for _ in metrics_data]).cpu().numpy()
        return np.sum(arr)
