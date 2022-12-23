"""
## Distribution-Balanced Loss

* Wu, Tong, Qingqiu Huang, Ziwei Liu, Yu Wang, and Dahua Lin. “Distribution-Balanced Loss for Multi-Label Classification
  in Long-Tailed Datasets.” In Computer Vision – ECCV 2020, edited by Andrea Vedaldi, Horst Bischof, Thomas Brox, and
  Jan-Michael Frahm, 162–78. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2020.
  https://doi.org/10.1007/978-3-030-58548-8_10.
* Huang, Yi, Buse Giledereli, Abdullatif Köksal, Arzucan Özgür, and Elif Ozkirimli. “Balancing Methods for Multi-Label
  Text Classification with Long-Tailed Class Distribution.” arXiv, October 15, 2021. http://arxiv.org/abs/2109.04712.

Code is adapted from the github repo belonging to the paper of Huang et al. https://github.com/Roche/BalancedLossNLP,
but pulled apart into different classes, which fitted this implementation better.

For training the type of loss can be set as a hyperparameter --loss_type.
"""

from argparse import ArgumentParser

import torch

from nucleotides.loss.base_loss import BaseLoss


class DistributionBalancedLoss(BaseLoss):
    def __init__(self, map_alpha=10.0, map_beta=0.2, map_gamma=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mapping function params
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_gamma = map_gamma

    def reweight(self, target):
        repeat_rate = torch.sum(target.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        # equation 4 Distribution-Balanced Loss for Multi-label Classification in Long-Tailed Datasets
        weight = (
                torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma))
                + self.map_alpha
        )
        return weight

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser = BaseLoss.add_args(parser)
        parser.add_argument("--map_alpha", type=float, default=10.0)
        parser.add_argument("--map_beta", type=float, default=0.2)
        parser.add_argument("--map_gamma", type=float, default=0.1)
        return parser
