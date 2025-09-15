import os
from contextlib import nullcontext
from typing import Callable, Optional

from tqdm import tqdm

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from .utils.loader import prepare_random_loader, prepare_sequential_loader


class Trainer:
    def __init__(self, model, optimizer: torch.optim.Optimizer, dtype=torch.bfloat16, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        self.ctx = nullcontext() if device == 'cpu' else autocast(device_type=device.split(':')[0], dtype=dtype)
        self.scaler = GradScaler(device, enabled=(dtype == torch.float16))

    def train_step_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Forward pass
        with self.ctx:
            _, loss = self.model(x, y)
        return loss

    @torch.no_grad()
    def evaluate_step_loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
        with self.ctx:
            logits, loss = self.model(x, y)
        return logits, loss.item()

    def backward_pass(self, grad_clip=1.0):
        if grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
    
    @torch.no_grad()
    def evaluation_step(self, data: np.ndarray, batch_size=16):
        # Calculate validation loss
        val_loss = sum(
            self.evaluate_step_loss(x, y)[1]
            for x, y in prepare_random_loader(data, batch_size, 10, self.model.config.block_size, self.device)
        ) / batch_size
        
        # Calculate perplexity
        rows_count = len(data) // self.model.config.block_size - 1
        seq_len = self.model.config.block_size - 1
        total_loss = sum(
            torch.nn.functional.cross_entropy(
                self.evaluate_step_loss(x, y)[0][:, :-1, :].contiguous().flatten(0, -2),
                y[:, :-1].reshape(-1), reduction='sum',
            ).item()
            for x, y in prepare_sequential_loader(data, batch_size, self.model.config.block_size, self.device)
        )
        ppl_val = np.exp(total_loss / rows_count / seq_len)

        return val_loss, ppl_val
    
    def train_step(self, data: np.ndarray, mini_batch_size: int, accum_steps: int, grad_clip=1.0):
        self.optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        for x, y in prepare_random_loader(data, mini_batch_size, accum_steps, self.model.config.block_size, self.device):
            loss = self.train_step_loss(x, y) / accum_steps
            train_loss += loss.item()
            self.scaler.scale(loss).backward()

        self.backward_pass(grad_clip)

        return train_loss
    
    def save_checkpoint(self, path: str, model: torch.nn.Module, step: int, best_val_loss: float, save_gradients: bool = False):
        if save_gradients:
            path_dir = os.path.dirname(path)
            path_name, path_ext = os.path.basename(path).rsplit('.', 1)

            grad_snapshot = {
                n: p.grad.detach().cpu()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            
            if len(grad_snapshot) > 0:
                torch.save(grad_snapshot, os.path.join(path_dir, f'{path_name}.grad.{path_ext}'))
                del grad_snapshot

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_num': step,
            'best_val_loss': best_val_loss,
            'config': model.config,
        }
        torch.save(checkpoint, path)
    
    def train(
            self,

            train_data: np.ndarray,
            val_data: np.ndarray,

            eval_interval: int = 100,
            steps: int = 1_000,
            start_step: int = 0,
            batch_size: int = 64,
            mini_batch_size: int = 16,
            grad_clip: float = 1.0,
            early_stop_patience: int = 0,
            
            sparsity_scheduler: Optional[Callable[[int], float]] = None,
            lr_scheduler: Optional[Callable[[int], float]] = None,

            best_model_dir: Optional[str] = None,
            checkpoint_dir: Optional[str] = None,
            model_save_interval: int = 0,
            save_gradients: bool = False,
            
            wandb=None,
    ):
        assert batch_size % mini_batch_size == 0, "batch_size must be divisible by mini_batch_size"
        
        accum_steps = batch_size // mini_batch_size

        train_data, val_data = train_data.astype(np.int64), val_data.astype(np.int64)

        val_history = []
        val_best = float('inf')

        self.model.train()

        current_lr, current_sparsity = float('nan'), float('nan')
        prev_val_loss, prev_ppl_val = float('nan'), float('nan')

        progress = tqdm(range(start_step, steps), desc="Training", unit="step", colour="green")
        for step in progress:
            if sparsity_scheduler is not None:
                current_sparsity = sparsity_scheduler(step)
                if wandb:
                    wandb.log({"sparsity_ratio": current_sparsity}, step=step)

            if lr_scheduler is not None:
                current_lr = lr_scheduler(step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                if wandb:
                    wandb.log({"lr": current_lr}, step=step)

            if step % eval_interval == 0:
                self.model.eval()
                
                prev_val_loss, prev_ppl_val = self.evaluation_step(val_data, mini_batch_size)
                
                self.model.train()
    
                progress.write(f"Step {step+1}/{steps}, Validation Loss: {prev_val_loss:.4f}, Validation Perplexity: {prev_ppl_val:.4f}")
                if wandb is not None:
                    wandb.log({"val/loss": prev_val_loss, "val/perplexity": prev_ppl_val}, step=step+1)

                if best_model_dir is not None and prev_val_loss < val_best:
                    val_best = prev_val_loss
                    self.save_checkpoint(
                        os.path.join(best_model_dir, 'best_model.pt'),
                        self.model,
                        step,
                        val_best,
                        save_gradients
                    )
                if checkpoint_dir is not None and step > 0 and step % model_save_interval == 0:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f'checkpoint_{step:06d}.pt'),
                        self.model,
                        step,
                        val_best,
                        save_gradients
                    )

                val_history.append(prev_ppl_val)
                if early_stop_patience > 0 and len(val_history) >= early_stop_patience:
                    recent = val_history[-early_stop_patience:]
                    if all(recent[i] >= recent[i-1] for i in range(1, early_stop_patience)):
                        progress.write(
                            f"Early stopping triggered: validation perplexity increased for the last {early_stop_patience} evals"
                        )
                        break

            train_loss = self.train_step(train_data, mini_batch_size, accum_steps, grad_clip)

            progress.set_postfix({
                "lr": current_lr,
                "sparsity": current_sparsity,
                "train_loss": train_loss,
                "val_loss": prev_val_loss,
                "val_ppl": prev_ppl_val,
            }, refresh=False)
            if wandb is not None:
                wandb.log({"train/loss": train_loss}, step=step+1)
