from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import METRIC_EPS

from nucleotides.metrics.confusion_matrix import multitask_confusion_matrix


class MultitaskBinnedAveragePrecision(Metric):
    def __init__(
            self,
            num_tasks: Optional[int] = None,
            prefix: Optional[str] = None,
            task_names: Optional[List[str]] = None,
            thresholds: Union[int, Tensor, List[float], None] = 20,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        if not num_tasks and not task_names:
            raise ValueError(
                "Arguments need to contain either 'num_tasks' or 'task_names'."
            )

        if task_names and not num_tasks:
            num_tasks = len(task_names)

        self.num_tasks = num_tasks
        self.prefix = prefix
        self.task_names = task_names
        if isinstance(thresholds, int):
            self.num_thresholds = thresholds
            thresholds = torch.linspace(0, 1.0, thresholds)
            self.register_buffer("thresholds", thresholds)
        elif thresholds is not None:
            if not isinstance(thresholds, (list, Tensor)):
                raise ValueError(
                    "Expected argument `thresholds` to either be an integer, list of floats or a tensor"
                )
            thresholds = (
                torch.tensor(thresholds) if isinstance(thresholds, list) else thresholds
            )
            self.num_thresholds = thresholds.numel()
            self.register_buffer("thresholds", thresholds)

        self.add_state(
            "confmat",
            default=torch.zeros(self.num_thresholds, 4, num_tasks),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """
        Args
            preds: (n_samples, n_tasks) tensor
            target: (n_samples, n_tasks) tensor
        """

        # Iterate one threshold at a time to conserve memory
        confusion_matrices = []
        for i in range(self.num_thresholds):
            conf_matrix_one_threshold = multitask_confusion_matrix(
                preds, target, self.thresholds[i]
            )
            confusion_matrices.append(conf_matrix_one_threshold)
        self.confmat += torch.stack(confusion_matrices)

    def _compute_average_precision(self):
        """Returns float tensor of size n_classes."""

        tps = self.confmat[:, 0]
        fps = self.confmat[:, 2]
        fns = self.confmat[:, 3]

        precisions = (tps + METRIC_EPS) / (tps + fps + METRIC_EPS)
        recalls = tps / (tps + fns + METRIC_EPS)

        # Need to guarantee that last precision=1 and recall=0, similar to precision_recall_curve
        t_ones = torch.ones(
            1, self.num_tasks, dtype=precisions.dtype, device=precisions.device
        )
        precisions = torch.cat([precisions, t_ones], dim=0)
        t_zeros = torch.zeros(
            1, self.num_tasks, dtype=recalls.dtype, device=recalls.device
        )
        recalls = torch.cat([recalls, t_zeros], dim=0)
        average_precision = -torch.sum(
            (recalls[1:] - recalls[:-1]) * precisions[:-1], dim=0
        )
        return average_precision

    def compute(self) -> dict:
        average_precision = self._compute_average_precision()
        metric_dict = {
            "avg_pr_auc": average_precision.mean(),
            "std_pr_auc": average_precision.std(),
            "max_pr_auc": average_precision.max(),
            "min_pr_auc": average_precision.min(),
        }
        return self._format_metric_names(metric_dict)

    def compute_for_tasks(self) -> pd.DataFrame:
        average_precision = self._compute_average_precision()
        with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
        ):
            dataframe = pd.DataFrame(
                average_precision.cpu().numpy(),
                index=self.task_names,
                columns=["pr_auc"],
            )

            if self.prefix:
                dataframe.columns = [
                    f"{self.prefix}_{column}" for column in dataframe.columns
                ]
            return dataframe

    def plot(self):
        """Returns float tensor of size n_classes."""
        tps = self.confmat[:, 0]
        fps = self.confmat[:, 2]
        fns = self.confmat[:, 3]

        precisions = ((tps + METRIC_EPS) / (tps + fps + METRIC_EPS)).cpu()
        recalls = (tps / (tps + fns + METRIC_EPS)).cpu()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlabel="precision", ylabel="recall")
        for task in range(self.num_tasks):
            precision = precisions[:, task]
            recall = recalls[:, task]
            ax = seaborn.lineplot(x=precision, y=recall, ax=ax)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return fig

    def _format_metric_names(self, metric_dict):
        if not self.prefix:
            return metric_dict
        return {f"{self.prefix}_{key}": value for key, value in metric_dict.items()}


if __name__ == "__main__":
    task_names = [f"task_{task}" for task in range(9)]

    dummy_pred = torch.rand(1024, 9)
    dummy_target = torch.bernoulli(torch.rand(1024, 9)).int()
    multitask_metrics = MultitaskBinnedAveragePrecision(
        thresholds=20, task_names=task_names, prefix="train"
    )
    multitask_metrics.update(dummy_pred, dummy_target)
    print(multitask_metrics.compute())
    print(multitask_metrics.compute_for_tasks())
    multitask_metrics.plot()
