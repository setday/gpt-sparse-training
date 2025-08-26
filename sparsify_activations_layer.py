from __future__ import annotations

import torch
from torch import nn
from typing import Optional

__all__ = [
    "LinearActivationsPruner",
    "replace_linears_with_pruner",
]

class LinearActivationsPruner(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity_type: Optional[str] = None,
        sparsity_ratio: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = float(sparsity_ratio)
        self.name = name

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def __repr__(self) -> str:  
        extra = (
            f", sparsity_type={self.sparsity_type}, sparsity_ratio={self.sparsity_ratio}"
            if self.sparsity_type is not None
            else ""
        )
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}{extra})"
        )

    @staticmethod
    def _compute_mask_rowwise(x: torch.Tensor, ratio: float) -> torch.Tensor:
        """0/1-маска для активаций, считается вдоль последней оси.
        Вырубает *k* наименьших |x| на каждом токене.
        """
        if ratio <= 0.0:
            return torch.ones_like(x, dtype=torch.bool)
        if ratio >= 1.0:
            return torch.zeros_like(x, dtype=torch.bool)

        *batch_dims, features = x.shape
        k = int(ratio * features)

        if k == 0:
            return torch.ones_like(x, dtype=torch.bool)
        if k >= features:
            return torch.zeros_like(x, dtype=torch.bool)

        abs_x = x.abs()
        threshold = torch.kthvalue(abs_x, k, dim=-1, keepdim=True).values
        return abs_x >= threshold

    @staticmethod
    def _compute_mask_global(w: torch.Tensor, ratio: float) -> torch.Tensor:
        """0/1-маска для весов – глобальный порог по всему тензору."""
        if ratio <= 0.0:
            return torch.ones_like(w, dtype=torch.bool)
        if ratio >= 1.0:
            return torch.zeros_like(w, dtype=torch.bool)

        flat = w.abs().flatten()
        k = int(ratio * flat.numel())
        if k == 0:
            return torch.ones_like(w, dtype=torch.bool)
        threshold = torch.kthvalue(flat, k).values
        return w.abs() >= threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        ratio = self.sparsity_ratio

        if self.sparsity_type == "masked-activations-layer":
            mask = self._compute_mask_rowwise(x, ratio).to(x.dtype)
            x = x * mask

        if self.sparsity_type == "masked-weights-layer":
            w_mask = self._compute_mask_rowwise(self.weight, ratio).to(self.weight.dtype)
            weight = self.weight * w_mask
        else:
            weight = self.weight

        out = torch.matmul(x, weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out

    def set_sparsity_ratio(self, sparsity_ratio: float) -> None:
        self.sparsity_ratio = float(sparsity_ratio)

    @classmethod
    def from_original(
        cls,
        orig_linear: nn.Linear,
        sparsity_type: Optional[str] = None,
        sparsity_ratio: float = 0.0,
        name: Optional[str] = None,
    ) -> "LinearActivationsPruner":
        pruner = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            bias=orig_linear.bias is not None,
            sparsity_type=sparsity_type,
            sparsity_ratio=sparsity_ratio,
            name=name,
        )
        pruner.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            pruner.bias.data.copy_(orig_linear.bias.data)
        return pruner

def set_sparsity_ratio(model: nn.Module, ratio: float):
    for m in model.modules():
        if isinstance(m, LinearActivationsPruner):
            m.set_sparsity_ratio(ratio)   # просто перезаписываем атрибут, его изменение не влияет на поведение оптимизатора


def replace_linears_with_pruner(
    module: nn.Module,
    sparsity_ratio: float,
    sparsity_type: str = "masked-activations-layer",
    mode: str = "all",  # "all", "exclude-first-last", or "custom"
    custom_slice: Optional[slice] = None,  # for "custom" mode
):
    linear_layers = []

    def collect_linears(m: nn.Module, prefix=""):
        for name, child in m.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                linear_layers.append((full_name, child))
            elif isinstance(child, nn.Module):
                collect_linears(child, prefix=full_name)

    collect_linears(module)

    if mode == "all":
        to_replace = set(name for name, _ in linear_layers)
    elif mode == "exclude-first-last" and len(linear_layers) > 2:
        to_replace = set(name for name, _ in linear_layers[1:-1])
    elif mode == "custom":
        if custom_slice is None:
            raise ValueError("custom_slice must be provided when mode='custom'")
        to_replace = set(name for name, _ in linear_layers[custom_slice])
    else:
        to_replace = set()

    def replace(m: nn.Module, prefix=""):
        for name, child in m.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and full_name in to_replace:
                pruner = LinearActivationsPruner.from_original(
                    child,
                    sparsity_type=sparsity_type,
                    sparsity_ratio=sparsity_ratio,
                    name=full_name,
                ).to(child.weight.device)
                setattr(m, name, pruner)

            elif isinstance(child, LinearActivationsPruner):
                child.set_sparsity_ratio(sparsity_ratio)
                child.sparsity_type = sparsity_type

            else:
                replace(child, prefix=full_name)

    replace(module)
