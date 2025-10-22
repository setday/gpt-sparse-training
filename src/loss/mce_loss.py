from torch import Tensor
from torch.nn import CrossEntropyLoss


class MCELossWrapper(CrossEntropyLoss):
    def __init__(self, weight: Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = "mean", label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(
        self, logits: Tensor, target: Tensor, **batch
    ) -> dict[str, Tensor]:
        loss = super().forward(logits.view(-1, logits.size(-1)), target.view(-1))

        return {"loss": loss}
