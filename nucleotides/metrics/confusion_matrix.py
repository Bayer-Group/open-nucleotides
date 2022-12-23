from typing import Any, Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
from torch import Tensor
from torchmetrics import Metric


class MultitaskMetrics(Metric):
    """Getting metrics for multitask bionary classification
       problems using torchmetrics.

       Examples:
        >>> dummy_pred = torch.rand(1024, 9)
        >>> dummy_target = torch.bernoulli(torch.rand(1024, 9)).int()
        >>> multitask_metrics = MultitaskMetrics(
        >>>    num_tasks=9, prefix="train", task_names=[f"task_{task}" for task in range(9)])

        Add some predictions to accumulate metrics:
        >>> multitask_metrics(dummy_pred, dummy_target)
        >>> print(multitask_metrics.confmat)
        tensor([[256., 261., 284., 248., 239., 247., 273., 242., 235.],
        [284., 258., 241., 269., 274., 242., 259., 272., 284.],
        [233., 253., 262., 243., 255., 261., 265., 254., 211.],
        [251., 252., 237., 264., 256., 274., 227., 256., 294.]])
        >>> print(multitask_metrics(dummy_pred, dummy_target))
        {'train_avg_true_positives': tensor(0.2479) 'train_avg_mcc': tensor(0.0133), ... }
        >>> print(multitask_metrics.compute_for_tasks())
                            train_true_positives  train_true_negatives  ...
        task_0              0.250000              0.277344              ...
        task_1              0.254883              0.251953              ...
        task_2              0.277344              0.235352              ...
        task_3              0.242188              0.262695              ...
        task_4              0.233398              0.267578              ...
        task_5              0.241211              0.236328              ...
        task_6              0.266602              0.252930              ...
        task_7              0.236328              0.265625              ...
        task_8              0.229492              0.277344              ...

    """

    def __init__(
            self,
            num_tasks: Optional[int] = None,
            prefix: Optional[str] = None,
            task_names: Optional[List[str]] = None,
            threshold: float = 0.5,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if not num_tasks and not task_names:
            raise ValueError(
                "Arguments need to contain either 'num_tasks' or 'task_names'."
            )

        if task_names and not num_tasks:
            num_tasks = len(task_names)

        self.add_state(
            "confmat", default=torch.zeros(4, num_tasks), dist_reduce_fx="sum"
        )
        self.prefix = prefix
        self.task_names = task_names
        self.threshold = threshold

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        confmat = multitask_confusion_matrix(preds, target, threshold=self.threshold)
        self.confmat += confmat

    def compute(self) -> Union[dict, pd.DataFrame]:
        """Computes matthews correlation coefficient."""
        return self._format_metric_names(scalar_report(self.confmat))

    def compute_for_tasks(self):
        with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
        ):
            metrics = per_task_report(
                confusion=self.confmat, prefix=self.prefix, task_names=self.task_names
            )
        return metrics

    def _format_metric_names(self, metric_dict):
        if not self.prefix:
            return metric_dict
        return {f"{self.prefix}_{key}": value for key, value in metric_dict.items()}

    def plot(self):
        """
        Creates a matplotlib figure containing a precision vs. recall
        and a specificity vs. recall scatter plot in which each dot is
        one task in of the multitask model.

        Returns: matplotlib.figure.Figure
        """
        with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
        ):
            metrics = per_task_report(
                confusion=self.confmat, task_names=self.task_names
            )
            prec = metrics["precision"]
            rec = metrics["recall"]
            spec = metrics["specificity"]

            fig = plt.figure()
            subplot = fig.add_subplot(121)
            subplot = seaborn.scatterplot(x=prec, y=rec, ax=subplot)
            subplot.set(xlabel="precision", ylabel="recall")
            subplot.set_xlim([0, 1])
            subplot.set_ylim([0, 1])

            subplot = fig.add_subplot(122)
            subplot = seaborn.scatterplot(x=spec, y=rec, ax=subplot)
            subplot.set(xlabel="specificity", ylabel="recall")
            subplot.set_xlim([0, 1])
            subplot.set_ylim([0, 1])

            return fig


def multitask_confusion_matrix(preds, target, threshold):
    preds = (preds > threshold).int()
    target = target.int()
    tp = (target * preds).sum(dim=0).to(torch.float32)
    tn = ((1 - target) * (1 - preds)).sum(dim=0).to(torch.float32)
    fp = ((1 - target) * preds).sum(dim=0).to(torch.float32)
    fn = (target * (1 - preds)).sum(dim=0).to(torch.float32)
    return torch.stack([tp, tn, fp, fn])


def mcc(confusion):
    tp = confusion[0]
    tn = confusion[1]
    fp = confusion[2]
    fn = confusion[3]
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator


def recall(confusion):
    tp = confusion[0]
    fn = confusion[3]
    return tp / (tp + fn)


def precision(confusion):
    tp = confusion[0]
    fp = confusion[2]
    return tp / (tp + fp)


def specificity(confusion):
    tn = confusion[1]
    fp = confusion[2]
    return tn / (tn + fp)


def balanced_accuracy(confusion):
    rec = recall(confusion)
    spec = specificity(confusion)
    return (rec + spec) / 2


def positive_rate(confusion):
    tp = confusion[0]
    fp = confusion[2]
    return (tp + fp) / confusion.sum(dim=0)


def support(confusion):
    tp = confusion[0]
    fn = confusion[3]
    return (tp + fn) / confusion.sum(dim=0)


def scalar_report(confusion):
    total = confusion.sum(dim=0)

    tp = confusion[0] / total
    tn = confusion[1] / total
    fp = confusion[2] / total
    fn = confusion[3] / total

    rec = recall(confusion)
    pre = precision(confusion)
    spec = specificity(confusion)

    balanced_acc = balanced_accuracy(confusion)

    sup = support(confusion)
    pos_rate = positive_rate(confusion)

    matthews = mcc(confusion)
    worst_index = argmin(matthews)
    best_index = argmax(matthews)
    median_index = argmedian(matthews)

    results = {
        "avg_true_positives": nanmean(tp),
        "avg_true_negatives": nanmean(tn),
        "avg_false_positives": nanmean(fp),
        "avg_false_negatives": nanmean(fn),
        "avg_recall": nanmean(rec),
        "avg_precision": nanmean(pre),
        "avg_specificity": nanmean(spec),
        "avg_balanced_accuracy": nanmean(balanced_acc),
        "avg_mcc": nanmean(matthews),
        "std_mcc": matthews.std(),
        "worst_model_recall": rec[worst_index],
        "worst_model_precision": pre[worst_index],
        "worst_model_mcc": matthews[worst_index],
        "best_model_recall": rec[best_index],
        "best_model_precision": pre[best_index],
        "best_model_mcc": matthews[best_index],
        "best_model_true_positives": tp[best_index],
        "best_model_true_negatives": tn[best_index],
        "best_model_false_positives": fp[best_index],
        "best_model_false_negatives": fn[best_index],
        "best_model_support": sup[best_index],
        "best_model_positive_rate": pos_rate[best_index],
        "median_model_recall": rec[median_index],
        "median_model_precision": pre[median_index],
        "median_model_mcc": matthews[median_index],
        "median_model_true_positives": tp[median_index],
        "median_model_true_negatives": tn[median_index],
        "median_model_false_positives": fp[median_index],
        "median_model_false_negatives": fn[median_index],
        "median_model_support": sup[median_index],
        "median_model_positive_rate": pos_rate[median_index],
    }
    return results


def per_task_report(confusion, prefix=None, task_names=None):
    total = confusion.sum(dim=0)

    dataframe = pd.DataFrame(
        (confusion / total).transpose(0, 1).cpu().numpy(),
        index=task_names,
        columns=[
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ],
    )
    dataframe["precision"] = precision(confusion).cpu().numpy()
    dataframe["recall"] = recall(confusion).cpu().numpy()
    dataframe["specificity"] = specificity(confusion).cpu().numpy()
    dataframe["balanced_accuracy"] = balanced_accuracy(confusion).cpu().numpy()
    dataframe["mcc"] = mcc(confusion).cpu().numpy()

    if prefix:
        dataframe.columns = [f"{prefix}_{column}" for column in dataframe.columns]
    return dataframe


def pr_scatter_plot(precisions, recalls):
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    subplot = seaborn.scatterplot(x=precisions, y=recalls, ax=subplot)
    subplot.set(xlabel="precision", ylabel="recall")
    subplot.set_xlim([0, 1])
    subplot.set_ylim([0, 1])
    return fig


def nanmean(tensor):
    """Average that of all values that are not NaN"""

    return tensor[~torch.isnan(tensor)].mean()


def argmax(tensor):
    """Argmax of all values that are not NaN"""

    tensor = tensor.clone()
    nan_mask = tensor != tensor
    tensor[nan_mask] = -1 * np.inf
    return tensor.argmax()


def argmin(tensor):
    """Argmax of all values that are not NaN"""

    tensor = tensor.clone()
    nan_mask = tensor != tensor
    tensor[nan_mask] = np.inf
    return tensor.argmin()


def argmedian(tensor):
    sort_object = tensor.sort()
    return sort_object.indices[len(sort_object.indices) // 2]
