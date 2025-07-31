from __future__ import annotations

import torch
from torch import nn
from typing import Optional

__all__ = [
    "LinearActivationsPruner",
    "replace_linears_with_pruner",
]


class LinearActivationsPruner(nn.Module):
    """Замена :class:`torch.nn.Linear` с возможностью разреживания.

    Поддерживаемые режимы:
    - ``None``                       – без разреживания
    - ``"masked-activations-layer"`` – обнуляем часть **активаций** (по признакам)
    - ``"masked-weights-layer"``     – обнуляем часть **весов** слоя
    """

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

        # Параметры слоя той же формы, что и у nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    # ---------------------------------------------------------------------
    # Pretty‑print helpers
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover – repr is for humans
        extra = (
            f", sparsity_type={self.sparsity_type}, sparsity_ratio={self.sparsity_ratio}"
            if self.sparsity_type is not None
            else ""
        )
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}{extra})"
        )

    # ------------------------------------------------------------------
    # Статические функции для вычисления масок
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mask_activation(x: torch.Tensor, ratio: float) -> torch.Tensor:
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
        # k-й наименьший вдоль последней оси
        threshold = torch.kthvalue(abs_x, k, dim=-1, keepdim=True).values
        return abs_x >= threshold

    @staticmethod
    def _compute_mask_weight(w: torch.Tensor, ratio: float) -> torch.Tensor:
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        ratio = self.sparsity_ratio

        # 1. Прореживаем активации (если требуется)
        if self.sparsity_type == "masked-activations-layer":
            mask = self._compute_mask_activation(x, ratio).to(x.dtype)
            x = x * mask

        # 2. Готовим веса (сырые или прореженные)
        if self.sparsity_type == "masked-weights-layer":
            w_mask = self._compute_mask_weight(self.weight, ratio).to(self.weight.dtype)
            weight = self.weight * w_mask
        else:
            weight = self.weight

        # 3. Линейное преобразование
        out = torch.matmul(x, weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out

    # ------------------------------------------------------------------
    def set_sparsity_ratio(self, sparsity_ratio: float) -> None:
        self.sparsity_ratio = float(sparsity_ratio)

    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Хелпер для массовой замены слоёв
# ---------------------------------------------------------------------------

def replace_linears_with_pruner(
    module: nn.Module,
    sparsity_ratio: float,
    sparsity_type: str = "masked-activations-layer",
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            pruner = LinearActivationsPruner.from_original(
                child,
                sparsity_type=sparsity_type,
                sparsity_ratio=sparsity_ratio,
                name=name,
            ).to(child.weight.device)
            setattr(module, name, pruner)
        elif isinstance(child, LinearActivationsPruner):
            child.set_sparsity_ratio(sparsity_ratio)
            child.sparsity_type = sparsity_type
        else:
            replace_linears_with_pruner(child, sparsity_ratio, sparsity_type)
