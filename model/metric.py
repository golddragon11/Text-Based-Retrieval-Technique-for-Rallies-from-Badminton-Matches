import torch
import numpy as np
import scipy
import scipy.stats
import itertools
from model.RetrievalRecall import retrieval_recall
from model.RetrievalPrecision import retrieval_precision
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix


class MulticlassMetric:
    def __init__(self, task, n_classes):
        assert task in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
        self.__name__ = task
        self.averagedMetric = None
        if task == "accuracy":
            self.metric = MulticlassAccuracy(num_classes=n_classes, average=None)
            self.averagedMetric = MulticlassAccuracy(num_classes=n_classes)
        elif task == "precision":
            self.metric = MulticlassPrecision(num_classes=n_classes, average=None)
            self.averagedMetric = MulticlassPrecision(num_classes=n_classes)
        elif task == "recall":
            self.metric = MulticlassRecall(num_classes=n_classes)
        elif task == "f1":
            self.metric = MulticlassF1Score(num_classes=n_classes)
        elif task == "confusion_matrix":
            self.metric = MulticlassConfusionMatrix(num_classes=n_classes)

    def __call__(self, output, target):
        self.metric.update(output, target)
        return self.metric.compute()

    def compute(self):
        if self.averagedMetric is not None:
            return {
                f'{self.__name__}': self.metric.compute() if self.__name__ != 'confusion_matrix' else self.metric.compute().int(),
                f'{self.__name__}_avg': self.averagedMetric.compute() if self.__name__ != 'confusion_matrix' else self.averagedMetric.compute().int()
            }
        else:
            return {
                f'{self.__name__}': self.metric.compute() if self.__name__ != 'confusion_matrix' else self.metric.compute().int()
            }

    def update(self, output, target):
        self.metric.update(output, target)
        if self.averagedMetric is not None:
            self.averagedMetric.update(output, target)

    def reset(self):
        self.metric.reset()
        if self.averagedMetric is not None:
            self.averagedMetric.reset()


# TODO: Add Median Rank & Mean Rank metrics
class MyRetrievalMetric:
    def __init__(self, task):
        self.__name__ = f"{task}_metrics"
        self.task = task

    def __call__(self, sims_dict, class_arr, eval_class_arr):
        metrics = {}
        sims = sims_dict[self.task]
        metrics.update(self.retrieval_metrics(sims, class_arr, eval_class_arr))
        return metrics

    def retrieval_metrics(self, sims, class_arr, eval_class_arr):
        target = np.zeros(sims.shape)
        class_arr = class_arr.detach().cpu().numpy()
        eval_class_arr = eval_class_arr.detach().cpu().numpy()
        for i, cls in enumerate(eval_class_arr):
            target[i][np.any(class_arr == cls, axis=1)] = 1

        metrics = {}
        target = torch.from_numpy(target)
        sims = torch.from_numpy(sims)
        for k in [1, 2, 3, 4, 5, 10, 25, 50]:
            metrics[f"R{k}"] = float(torch.mean(retrieval_recall(sims, target, k=k, num_tasks=eval_class_arr.size))) * 100
            metrics[f"P{k}"] = float(torch.mean(retrieval_precision(sims, target, k=k, num_tasks=eval_class_arr.size))) * 100
        metrics['Average Positive Targets'] = float(torch.mean(torch.sum(target, dim=1)))

        return metrics

    def __repr__(self):
        return f"{self.task}_metrics"
