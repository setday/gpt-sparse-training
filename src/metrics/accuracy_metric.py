from typing import List

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_top_acc


class AccuracyAtOne(BaseMetric):
    def __init__(
            self,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, target: Tensor, **batch):
        return calc_top_acc(target, logits, top_k=1)
    
class AccuracyAtFive(BaseMetric):
    def __init__(
            self,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, target: Tensor, **batch):
        return calc_top_acc(target, logits, top_k=5)
    
class FullAccuracy(BaseMetric):
    def __init__(
            self,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, target: Tensor, **batch):
        predicted = logits.argmax(dim=-1)
        return (predicted == target).float().mean()
