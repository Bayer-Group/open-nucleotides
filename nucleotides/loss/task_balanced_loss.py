from argparse import ArgumentParser

import torch


class TaskBalancedLoss(torch.nn.Module):
    def __init__(
            self,
            positive_weight: torch.Tensor,
            positive_weight_exponent: float = 1,
            scale_loss_magnitude: bool = True,
            loss_function=None,
            *args,
            **kwargs
    ):
        """
        Rescale the weights for the positive and negative class so that all
        the positive weights are on 1 (for exponent == 0) and the negatives
        scales down by 1 / pos_weight or the other way around: negatives are
        weight 1 and the positives are scaled by pos_weight.
        In a single task model this should not have any influence. In a
        multitask model it influences the balance between the tasks though.
        Args:
            positive_weight: positive weights for tasks
            positive_weight_exponent: makes sense to be between 0 to 1.
            keep_magnitude: eventhough rebalancing is happening, scale in
                such a way that the final average magnitude of the loss
                is the same as when balancing would not have happened.
        """
        super().__init__()

        self.register_buffer("pos_weight", positive_weight)
        self.register_buffer(
            "adjusted_positive_weight", positive_weight ** positive_weight_exponent
        )
        self.register_buffer(
            "adjusted_negative_weight", self.adjusted_positive_weight / self.pos_weight
        )

        self.loss_function = loss_function or torch.nn.BCEWithLogitsLoss(
            reduction="none"
        )
        self.keep_magnitude = scale_loss_magnitude

    def forward(self, pred, target):
        positives = target
        negatives = (~target.bool()).int()
        weights = (
                positives * self.adjusted_positive_weight
                + negatives * self.adjusted_negative_weight
        )
        bce_loss = self.loss_function(pred, target)
        adjusted_loss = (bce_loss * weights).mean()

        if self.keep_magnitude:
            positives = target
            negatives = (~target.bool()).int()
            weights = positives * self.pos_weight + negatives * 1
            plain_loss = (bce_loss * weights).mean()
            magnitude_reduction = (adjusted_loss / plain_loss).detach()
        else:
            magnitude_reduction = 1
        return adjusted_loss / magnitude_reduction

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--positive_weight_exponent", type=float, default=1.0)
        parser.add_argument("--scale_loss_magnitude", type=bool, default=True)
        return parser


if __name__ == "__main__":
    positive_weight = torch.Tensor([0.8, 2, 10, 4])
    loss = TaskBalancedLoss(positive_weight, 1.0)
    dummy_target = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]]).float()
    dummy_pred = torch.tensor([[0.5, 0, 0.1, 0.2], [0, 0.9, 0, 0.7], [0.1, 0, 1, 0.2]])
    print(loss(dummy_pred, dummy_target))
