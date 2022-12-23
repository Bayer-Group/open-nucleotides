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

from nucleotides.loss.base_loss import BaseLoss
from nucleotides.loss.class_balanced_loss import ClassBalancedLoss
from nucleotides.loss.distribution_balanced_loss import DistributionBalancedLoss
from nucleotides.loss.task_balanced_loss import TaskBalancedLoss

loss_types = {
    "task_balanced": TaskBalancedLoss,
    "class_balanced": ClassBalancedLoss,
    "distribution_balanced": DistributionBalancedLoss,
    "base": BaseLoss,
}


def get_loss(hparams):
    hparam_dict = vars(hparams)
    return loss_types[hparam_dict.get("loss_type", "task_balanced")](**hparam_dict)


def add_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # parser.add_argument(
    #     "--positive_weight", type=float, default=get_positive_weight()
    # )
    parser.add_argument("--loss_type", type=str, default="task_balanced")
    for loss_type in loss_types.values():
        parser = loss_type.add_args(parser)
    return parser
