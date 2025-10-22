from __future__ import annotations

from typing import Literal, Optional, List

import torch
from torch import nn

__all__ = [
    "LinearPruner",
    "replace_linears_with_pruner",
]

SparsityType = Literal["masked-activations-layer", "masked-weights-layer", "none"]
SparsityScale = Literal["rowwise", "global", "fixed", "none"]

class LinearPruner(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity_type: Optional[SparsityType] = None,
        sparsity_scale: Optional[SparsityScale] = None,
        sparsity_ratio: float = 0.0,
        name: Optional[str] = None,

        debug_info: bool = False,
        l1_calculation: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type if sparsity_type != "none" else None
        self.sparsity_scale = sparsity_scale if sparsity_scale != "none" else None
        self.sparsity_ratio = float(sparsity_ratio)
        self.name = name

        self.debug_info = debug_info
        self.l1_calculation = l1_calculation

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.in_l1, self.out_l1 = None, None
        self.params_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.real_weight_sparsity, self.real_input_sparsity, self.real_output_sparsity = None, None, None
        self.real_weight_positive, self.real_input_positive, self.real_output_positive = None, None, None
        self.real_weight_negative, self.real_input_negative, self.real_output_negative = None, None, None
        self.threshold = None

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
    def _compute_threshold_rowwise(x: torch.Tensor, ratio: float) -> torch.Tensor:
        """0/1-маска для активаций, считается вдоль последней оси.
        Вырубает *k* наименьших |x| на каждом токене.
        """
        features = x.shape[-1]
        k = int(ratio * features)

        if k <= 0:
            return x.min(dim=-1, keepdim=True).values - 0.001
        if k >= features:
            return x.max(dim=-1, keepdim=True).values + 0.001

        return torch.kthvalue(x, k, dim=-1, keepdim=True).values

    @staticmethod
    def _compute_threshold_global(w: torch.Tensor, ratio: float) -> torch.Tensor:
        """0/1-маска для весов – глобальный порог по всему тензору."""
        flat = w.flatten()

        features = flat.numel()
        k = int(ratio * features)
        
        if k <= 0:
            return flat.min() - 0.001
        if k >= features:
            return flat.max() + 0.001

        return torch.kthvalue(flat, k).values

    @staticmethod
    def compute_threshold(x: torch.Tensor, scale: SparsityScale, ratio: float) -> torch.Tensor:
        if scale == "rowwise":
            return LinearPruner._compute_threshold_rowwise(x, ratio).detach()
        if scale == "global":
            return LinearPruner._compute_threshold_global(x, ratio).detach()
        if scale == "fixed":
            return torch.tensor(ratio, device=x.device)

        return torch.tensor(float('-inf'), device=x.device)  # no sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x_ = x
        if self.sparsity_scale is not None and self.sparsity_type == "masked-activations-layer":
            x_abs = x_.abs()
            self.threshold = LinearPruner.compute_threshold(x_abs, self.sparsity_scale, self.sparsity_ratio)
            a_mask = x_abs >= self.threshold
            x_ = x_ * a_mask.to(x_.dtype)

        weight_ = self.weight
        if self.sparsity_scale is not None and self.sparsity_type == "masked-weights-layer":
            weight_abs = weight_.abs()
            self.threshold = LinearPruner.compute_threshold(weight_abs, self.sparsity_scale, self.sparsity_ratio)
            w_mask = weight_abs >= self.threshold
            weight_ = weight_ * w_mask.to(weight_.dtype)

        out = torch.matmul(x_, weight_.t())
        if self.bias is not None:
            out = out + self.bias

        if self.l1_calculation:
            self.in_l1 = x.abs().mean()
            self.out_l1 = out.abs().mean()

        if self.debug_info:
            self.real_input_sparsity = (x == 0).float().mean().item()
            self.real_output_sparsity = (out == 0).float().mean().item()
            self.real_weight_sparsity = (self.weight == 0).float().mean().item()
            self.real_input_positive = (x > 0).float().mean().item()
            self.real_output_positive = (out > 0).float().mean().item()
            self.real_weight_positive = (self.weight > 0).float().mean().item()
            self.real_input_negative = 1.0 - self.real_input_positive - self.real_input_sparsity
            self.real_output_negative = 1.0 - self.real_output_positive - self.real_output_sparsity
            self.real_weight_negative = 1.0 - self.real_weight_positive - self.real_weight_sparsity

        return out
    
    def get_l1_loss(self, l1_target: Literal["weight", "input", "output"] = "weight") -> torch.Tensor:
        if l1_target == "weight":
            return self.weight.abs().mean()
        elif l1_target == "input":
            if self.in_l1 is None:
                raise ValueError("in_l1 is not set. Run a forward pass first and check l1_calculation option.")
            return self.in_l1
        elif l1_target == "output":
            if self.out_l1 is None:
                raise ValueError("out_l1 is not set. Run a forward pass first and check l1_calculation option.")
            return self.out_l1
        elif l1_target == "none" or l1_target is None:
            return torch.tensor(0.0, device=self.weight.device)
        else:
            raise ValueError("l1_target must be 'weight', 'input', or 'output'")
        
    def get_real_sparsities_info(self) -> tuple[float, float, float, float, float, float, float, float, float]:
        assert all(v is not None for v in [
                self.real_weight_sparsity,
                self.real_input_sparsity,
                self.real_output_sparsity,
                self.real_weight_positive,
                self.real_input_positive,
                self.real_output_positive,
                self.real_weight_negative,
                self.real_input_negative,
                self.real_output_negative
        ]), "real sparsities are not set. Run a forward pass first and check debug_info option."

        return self.real_weight_sparsity, self.real_input_sparsity, self.real_output_sparsity,\
                self.real_weight_positive, self.real_input_positive, self.real_output_positive,\
                self.real_weight_negative, self.real_input_negative, self.real_output_negative
    
    def get_threshold_info(self) -> tuple[float, float, float]:
        assert self.threshold is not None, "threshold is not set. Run a forward pass first."

        return self.threshold.min().item(), self.threshold.mean().item(), self.threshold.max().item()

    def set_sparsity_ratio(self, sparsity_ratio: float) -> None:
        self.sparsity_ratio = float(sparsity_ratio)

    @classmethod
    def from_original(
        cls,
        orig_linear: nn.Linear,
        sparsity_type: Optional[SparsityType] = None,
        sparsity_ratio: float = 0.0,
        sparsity_scale: Optional[Literal["rowwise", "global"]] = None,
        name: Optional[str] = None,
    ) -> LinearPruner:
        pruner = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            bias=orig_linear.bias is not None,
            sparsity_type=sparsity_type,
            sparsity_scale=sparsity_scale,
            sparsity_ratio=sparsity_ratio,
            name=name,
        )
        pruner.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            pruner.bias.data.copy_(orig_linear.bias.data)
        return pruner

def set_sparsity_ratio(model: nn.Module, ratio: float):
    for m in model.modules():
        if isinstance(m, LinearPruner):
            m.set_sparsity_ratio(ratio)   # просто перезаписываем атрибут, его изменение не влияет на поведение оптимизатора


def replace_linears_with_pruner(
    module: nn.Module,
    sparsity_ratio: float,
    sparsity_type: Optional[SparsityType] = "masked-activations-layer",
    sparsity_scale: Optional[SparsityScale] = "rowwise",
    mode: str = "all",  # "all", "exclude-first-last", or "custom"
    custom_slice: Optional[slice] = None,  # for "custom" mode
) -> List[LinearPruner]:
    sparsity_type = sparsity_type if sparsity_type != "none" else None
    sparsity_scale = sparsity_scale if sparsity_scale != "none" else None

    linear_layers = [
        (name, layer)
        for name, layer in module.named_modules()
        if isinstance(layer, nn.Linear) or isinstance(layer, LinearPruner)
    ]

    if mode == "all":
        to_replace = set(name for name, _ in linear_layers)
    elif mode == "exclude-first-last" and len(linear_layers) > 2:
        to_replace = set(name for name, _ in linear_layers[1:-1])
    elif mode == "custom":
        if custom_slice is None:
            raise ValueError("custom_slice must be provided when mode='custom'")
        to_replace = set(name for name, _ in linear_layers[custom_slice])
    else:
        raise ValueError("mode must be 'all', 'exclude-first-last', or 'custom'")

    resulting_layers = []
    
    for name, layer in module.named_modules():
        for child_name, child in layer.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and full_name in to_replace:
                pruner = LinearPruner.from_original(
                    child,
                    sparsity_type=sparsity_type,
                    sparsity_ratio=sparsity_ratio,
                    sparsity_scale=sparsity_scale,
                    name=full_name,
                ).to(child.weight.device)
                setattr(layer, child_name, pruner)
                resulting_layers.append(pruner)

            elif isinstance(child, LinearPruner):
                child.set_sparsity_ratio(sparsity_ratio)
                child.sparsity_type = sparsity_type
                child.sparsity_scale = sparsity_scale
                resulting_layers.append(child)

    return resulting_layers
