import torch
from torch import nn
from typing import Optional

__all__ = [
    "LinearActivationsPruner",
    "replace_linears_with_pruner",
]


class LinearActivationsPruner(nn.Module):
    """A drop‑in replacement for :class:`torch.nn.Linear` with on‑the‑fly pruning.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        If set to ``False``, the layer will not learn an additive bias.
    sparsity_type : {None, "masked-activations-layer", "masked-weights-layer"}
        Which tensor to prune. ``None`` disables pruning.
    sparsity_ratio : float, optional
        Fraction (0–1) of elements to set to zero.
    name : str, optional
        An arbitrary identifier (helpful for debugging / logging).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity_type: Optional[str] = None,
        sparsity_ratio: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = sparsity_ratio or 0.0
        self.name = name

        # Weight / bias follow the same shape conventions as nn.Linear
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

    # ---------------------------------------------------------------------
    # Pruning helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_mask(tensor: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """Return a 0/1 mask such that the given fraction of smallest |values| is zero."""
        if sparsity_ratio <= 0.0:
            return torch.ones_like(tensor, dtype=torch.bool)
        if sparsity_ratio >= 1.0:
            return torch.zeros_like(tensor, dtype=torch.bool)

        # Flatten so we can apply one global threshold per tensor
        flat = torch.abs(tensor).flatten()
        k = int(sparsity_ratio * flat.numel())
        if k == 0:
            return torch.ones_like(tensor, dtype=torch.bool)
        # ``torch.kthvalue`` gives the *k‑th* smallest (1‑indexed)
        threshold = torch.kthvalue(flat, k).values  # shape: []
        return torch.abs(tensor) >= threshold  # broadcast back to original shape

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        mask_ratio = float(self.sparsity_ratio or 0.0)

        # 1. Prune activations if requested
        if self.sparsity_type == "masked-activations-layer":
            mask = self._compute_mask(x, mask_ratio).to(x.dtype)
            x = x * mask

        # 2. Prepare weight (pruned or raw)
        if self.sparsity_type == "masked-weights-layer":
            w_mask = self._compute_mask(self.weight, mask_ratio).to(self.weight.dtype)
            weight = self.weight * w_mask
        else:
            weight = self.weight

        # 3. Linear transform
        out = torch.matmul(x, weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def set_sparsity_ratio(self, sparsity_ratio: float) -> None:
        """Dynamically update the sparsity ratio (for both weight and activation modes)."""
        self.sparsity_ratio = float(sparsity_ratio)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_original(
        cls,
        orig_linear: nn.Linear,
        sparsity_type: Optional[str] = None,
        sparsity_ratio: Optional[float] = None,
        name: Optional[str] = None,
    ) -> "LinearActivationsPruner":
        """Create a pruner layer by cloning an existing :class:`nn.Linear`."""
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
# Module‑level helper to swap all linears in a model or sub‑module
# ---------------------------------------------------------------------------

def replace_linears_with_pruner(
    module: nn.Module,
    sparsity_ratio: float,
    sparsity_type: str = "masked-activations-layer",
):
    """Recursively replace every :class:`nn.Linear` with a pruner counterpart.

    Examples
    --------
    >>> replace_linears_with_pruner(model, sparsity_ratio=0.15,
    ...                             sparsity_type="masked-weights-layer")
    """

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
            # Recurse into nested sub‑modules
            replace_linears_with_pruner(child, sparsity_ratio, sparsity_type)
