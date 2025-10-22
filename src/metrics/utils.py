from typing import List

from torch import Tensor
from timm.utils import accuracy


def calc_top_acc(targets: List[int] | Tensor, preds: List[int] | Tensor, top_k: int = 1) -> float:
    """
    Calculate the top-k accuracy.
    """
    acc = accuracy(preds, targets, topk=(top_k,))[0]
    if isinstance(acc, Tensor):
        return acc.item()
    return acc
