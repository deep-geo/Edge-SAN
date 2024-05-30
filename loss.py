import torch
from torch import nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask, weights=None):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        if weights is not None:
            loss_pos *= weights.view(weights.shape[0], 1, 1, 1)
            loss_neg *= weights.view(weights.shape[0], 1, 1, 1)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask, weights=None):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)

        intersection = p * mask
        union = p + mask

        if weights is not None:
            intersection *= weights.view(weights.shape[0], 1, 1, 1)
            union *= weights.view(weights.shape[0], 1, 1, 1)

        dice_loss = ((2.0 * intersection.sum() + self.smooth) /
                     (union.sum() + self.smooth))

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou, weights=None):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, \
            "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask, dim=(2, 3))
        union = torch.sum(p + ground_truth_mask, dim=(2, 3)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        if weights is None:
            iou_loss = torch.mean((iou - pred_iou) ** 2)
        else:
            iou_loss = torch.mean(
                (iou - pred_iou) ** 2 * weights.view(weights.shape[0], 1, 1, 1)
            )
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):

    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou, weights=None):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask, weights)
        dice_loss =self.dice_loss(pred, mask, weights)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou, weights)
        loss = loss1 + loss2 * self.iou_scale
        return loss


class BCE_Loss(nn.Module):

    def __init__(self):
        super(BCE_Loss, self).__init__()

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W] - Predictions as logits
        mask: [B, 1, H, W] - Ground truth masks
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)  # Apply sigmoid to convert logits to probabilities

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(p, mask, reduction='mean')
        return bce_loss


class BCE_Diceloss_IoULoss(nn.Module):

    def __init__(self, weight=1.0, iou_scale=1.0):
        super(BCE_Diceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.bce_loss = BCE_Loss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        BCE_loss = self.BCE_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = BCE_loss + dice_loss
        #loss2 = self.maskiou_loss(pred, mask, pred_iou)
        #loss = loss1 #+ loss2 * self.iou_scale
        return loss1
