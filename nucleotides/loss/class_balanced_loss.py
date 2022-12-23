"""
## Class-Balanced Loss

* Cui, Yin, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. “Class-Balanced Loss Based on Effective Number of
  Samples.” In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9260–69, 2019.
  https://doi.org/10.1109/CVPR.2019.00949.
* Huang, Yi, Buse Giledereli, Abdullatif Köksal, Arzucan Özgür, and Elif Ozkirimli. “Balancing Methods for Multi-Label
  Text Classification with Long-Tailed Class Distribution.” arXiv, October 15, 2021. http://arxiv.org/abs/2109.04712.

Code is adapted from the github repo belonging to the paper of Huang et al. https://github.com/Roche/BalancedLossNLP,
but pulled apart into different classes, which fitted this implementation better.

For training the type of loss can be set as a hyperparameter --loss_type.
"""

from argparse import ArgumentParser

import torch
from nucleotides.loss.base_loss import BaseLoss


class ClassBalancedLoss(BaseLoss):
    def __init__(
            self,
            class_balance_beta=0.9,
            class_balance_mode="by_class",  # "average_w", 'by_class', 'average_n', 'average_w', 'min_n'
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_balance_beta = class_balance_beta
        self.class_balance_mode = class_balance_mode

    def reweight(self, target):
        if "by_class" in self.class_balance_mode:
            weight = torch.tensor((1 - self.class_balance_beta)) / (  # .cuda()
                    1 - torch.pow(self.class_balance_beta, self.class_freq)
            )  # .cuda()
        elif "average_n" in self.class_balance_mode:
            avg_n = torch.sum(
                target * self.class_freq, dim=1, keepdim=True
            ) / torch.sum(target, dim=1, keepdim=True)
            weight = (
                    torch.tensor((1 - self.class_balance_beta))  # .cuda()
                    / (1 - torch.pow(self.class_balance_beta, avg_n)).cuda()
            )
        elif "average_w" in self.class_balance_mode:
            weight_ = torch.tensor((1 - self.class_balance_beta)) / (  # .cuda()
                    1 - torch.pow(self.class_balance_beta, self.class_freq)
            )  # .cuda()
            weight = torch.sum(target * weight_, dim=1, keepdim=True) / torch.sum(
                target, dim=1, keepdim=True
            )
        elif "min_n" in self.class_balance_mode:
            min_n, _ = torch.min(
                target * self.class_freq + (1 - target) * 100000,
                dim=1,
                keepdim=True,
            )
            weight = torch.tensor((1 - self.class_balance_beta)) / (  # .cuda()
                    1 - torch.pow(self.class_balance_beta, min_n)
            )  # .cuda()
        else:
            raise NameError
        return weight

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser = BaseLoss.add_args(parser)
        parser.add_argument("--class_balance_beta", type=float, default=0.9)
        parser.add_argument("--class_balance_mode", type=str, default="by_class")
        return parser
