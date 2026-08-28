#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
losses_metrics.py

- BCE + Dice loss
- IoU / Precision / Recall / F1 の計算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss + Dice Loss の和
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : [B,1,H,W]
        targets: [B,1,H,W], 0/1
        """
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dice = dice_loss_from_probs(probs, targets)

        return self.bce_weight * bce + self.dice_weight * dice


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Sigmoid 済みの確率と 0/1 ターゲットから Dice Loss を計算
    """
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    loss = 1.0 - dice
    return loss.mean()


def batch_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    ミニバッチ単位の IoU （鏡クラス）を計算して平均を返す（train の進捗表示用）
    logits : [B,1,H,W]
    targets: [B,1,H,W], 0/1
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def accumulate_confusion_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresh: float = 0.5,
) -> tuple:
    """
    logits,targets から TP,FP,FN を返す（val エポック総和用）
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    return tp, fp, fn


def compute_iou_precision_recall_f1(tp: int, fp: int, fn: int, eps: float = 1e-6) -> dict:
    """
    累積 TP,FP,FN から IoU, Precision, Recall, F1 を計算
    """
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return dict(
        iou=iou,
        precision=precision,
        recall=recall,
        f1=f1,
    )
