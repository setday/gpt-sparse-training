from pathlib import Path

import pandas as pd

from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_top_acc
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, zero_grad=True, update=True):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train and zero_grad:
            self.optimizer.zero_grad()

        logits = self.model(batch["image"])
        batch.update({"logits": logits})

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            
            if update:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        metrics_split = "train" if self.is_train else "inference"
        for met in self.metrics[metrics_split]:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode != "train":  # the method is called only every self.log_step steps
            # Log Stuff
            self.log_predictions(**batch)

    def log_predictions(
        self, image, logits, target, image_path, examples_to_log=10, **batch
    ):
        preds = logits.argmax(dim=-1).cpu().numpy()

        self.writer.add_scalars({
            "top_1_acc": calc_top_acc(target, logits, top_k=1),
            "top_5_acc": calc_top_acc(target, logits, top_k=5),
        })

        image_paths = [Path(img_path).name if image_path is not None else str(i) for i, img_path in enumerate(image_path)]

        rows = {
            img_path: {
                "target": trg.item(),
                "predictions": pred,
            }
            for pred, trg, img_path in list(zip(preds, target, image_paths))[:examples_to_log]
        }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
