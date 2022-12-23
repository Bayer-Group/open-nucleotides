"""
## Loss Functions

Different losses are implemented in nucleotides/loss . For focal, "class balanced"
and "distribution balanced" losses, and another interesting paper about random weighting check:

* Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. “Focal Loss for Dense Object Detection.” In
  2017 IEEE International Conference on Computer Vision (ICCV), 2999–3007, 2017. https://doi.org/10.1109/ICCV.2017.324.
* Cui, Yin, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. “Class-Balanced Loss Based on Effective Number of
  Samples.” In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9260–69, 2019.
  https://doi.org/10.1109/CVPR.2019.00949.
* Wu, Tong, Qingqiu Huang, Ziwei Liu, Yu Wang, and Dahua Lin. “Distribution-Balanced Loss for Multi-Label Classification
  in Long-Tailed Datasets.” In Computer Vision – ECCV 2020, edited by Andrea Vedaldi, Horst Bischof, Thomas Brox, and
  Jan-Michael Frahm, 162–78. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2020.
  https://doi.org/10.1007/978-3-030-58548-8_10.
* Huang, Yi, Buse Giledereli, Abdullatif Köksal, Arzucan Özgür, and Elif Ozkirimli. “Balancing Methods for Multi-Label
  Text Classification with Long-Tailed Class Distribution.” arXiv, October 15, 2021. http://arxiv.org/abs/2109.04712.
* Lin, Baijiong, Feiyang Ye, Yu Zhang, and Ivor W. Tsang. “Reasonable Effectiveness of Random Weighting: A Litmus Test
  for Multi-Task Learning.” arXiv, July 27, 2022. http://arxiv.org/abs/2111.10603.

Code is adapted from the github repo belonging to the paper of Huang et al. https://github.com/Roche/BalancedLossNLP,
but pulled apart into different classes, which fitted this implementation better.

For training the type of loss can be set as a hyperparameter --loss_type.
"""

from argparse import ArgumentParser

import torch

import torch.nn.functional as F


class BaseLoss(torch.nn.Module):
    def __init__(
            self,
            reduction="mean",
            weight_norm=None,  # None, 'by_instance', 'by_batch'
            focal_alpha=None,
            focal_gamma=None,
            logit_reg_neg_scale=1.0,  # 0.5,
            logit_reg_init_bias=0.0,  # 0.1,
            positive_weight=None,
            train_num=1,
            *args,
            **kwargs
    ):
        super().__init__()

        self.reduction = reduction
        self.cls_criterion = binary_cross_entropy

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        class_freq = (1 / (positive_weight + 1)).float() * train_num

        self.register_buffer("class_freq", torch.Tensor(class_freq).float())

        # regularization params
        self.use_logit_reg = (
                logit_reg_neg_scale is not None and logit_reg_init_bias is not None
        )
        self.neg_scale = logit_reg_neg_scale or 1.0
        init_bias = logit_reg_init_bias or 0.0
        self.register_buffer(
            "init_bias", -torch.log(train_num / self.class_freq - 1) * init_bias
        )
        self.register_buffer("freq_inv", 1 / self.class_freq)
        self.register_buffer("propotion_inv", train_num / self.class_freq)

    def forward(self, logits, target, avg_factor=None, **kwargs):

        weight = self.reweight(target.float())

        if self.weight_norm == "by_instance":
            max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
            weight = weight / max_by_instance
        elif self.weight_norm == "by_batch":
            weight = weight / torch.max(weight)

        logits, weight = self.logit_reg_functions(
            logits=logits, target=target.float(), weight=weight
        )

        if self.focal_alpha and self.focal_gamma:
            loss = focal_loss(
                logits=logits,
                target=target,
                cls_criterion=self.cls_criterion,
                weight=weight,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction=self.reduction,
                avg_factor=avg_factor,
            )
        else:
            loss = self.cls_criterion(
                logits, target.float(), weight, reduction=self.reduction
            )

        return loss

    def logit_reg_functions(self, logits, target, weight=None):
        if not self.use_logit_reg:
            return logits, weight
        logits += self.init_bias
        if self.neg_scale:
            logits = logits * (1 - target) * self.neg_scale + logits * target
            if weight is not None:
                weight = weight / self.neg_scale * (1 - target) + weight * target
        return logits, weight

    def reweight(self, target):
        """Function to overwrite for more
        specific loss weighting.
        """
        raise NotImplementedError

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--reduction", type=str, default="mean")
        parser.add_argument("--weight_norm", type=str, default=None)
        parser.add_argument("--focal_alpha", type=float, default=None)
        parser.add_argument("--focal_gamma", type=float, default=None)
        parser.add_argument("--logit_reg_neg_scale", type=float, default=1.0)
        parser.add_argument("--logit_reg_init_bias", type=float, default=0.0)
        parser.add_argument("--train_num", type=float, default=1)
        return parser


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None):
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction="none"
    )
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def focal_loss(
        logits, target, cls_criterion, weight, alpha, gamma, reduction, avg_factor=None
):
    logpt = cls_criterion(
        logits.clone(),
        target,
        weight=None,
        reduction="none",
        avg_factor=avg_factor,
    )
    # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
    pt = torch.exp(-logpt)
    wtloss = cls_criterion(logits, target.float(), weight=weight, reduction="none")
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    loss = alpha_t * ((1 - pt) ** gamma) * wtloss
    loss = reduce_loss(loss, reduction)
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    if not reduction in (None, "mean", "sum"):
        raise ValueError(f"Argument reduction should be one of None, 'mean' or 'sum'.")

    if reduction is None:
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
