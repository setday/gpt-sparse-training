"""
This training script runs on a single GPU (or CPU) only. All logic related to
Distributed Data Parallel (DDP) has been removed for simplicity.

Examples:
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a GPT‑2 (124 M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' | 'resume' | 'gpt2*'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # simulates larger batch sizes
batch_size = 12  # micro‑batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # use bias inside LayerNorm and Linear layers?
# AdamW optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600_000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning‑rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # warm‑up steps
lr_decay_iters = 600_000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # ~= learning_rate/10 per Chinchilla
sparsity_ratio = 0.0
sparsity_type = "masked-activations-layer"

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 'cpu', 'cuda', 'cuda:0', etc.
dtype = (
    'bfloat16'
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else 'float16'
)  # 'float32', 'bfloat16', or 'float16'
compile = True  # use PyTorch 2.0 compile to speed up training
# -----------------------------------------------------------------------------
# Allow command‑line or config‑file overrides via configurator.py
config_keys = [
    k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
exec(open('configurator.py').read())  # may overwrite globals
config = {k: globals()[k] for k in config_keys}  # handy for logging
# -----------------------------------------------------------------------------

# Set up reproducibility and I/O
os.makedirs(out_dir, exist_ok=True)
seed_offset = 0

torch.manual_seed(1337 + seed_offset)

# Enable TF32 where available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Autocast context and dtype handling
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = (
    nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device.split(':')[0], dtype=ptdtype)
)

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Poor‑man's data loader -------------------------------------------------------

data_dir = os.path.join('data', dataset)


def get_batch(split):
    """Return one batch of data as (x, y) tensors."""
    path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix
    ])
    if device.startswith('cuda'):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Model setup ------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from dataset metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Build model
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT‑2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, sparsity_ratio, sparsity_type)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
        
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, sparsity_ratio, sparsity_type)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

print("======================================")
print("MODEL:")
print(model)
print("======================================")

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type=device.split(':')[0])
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free mem

# Optional Torch 2.0 compile
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

raw_model = model  # for MFU calculations

# ----------------------------------------------------------------------------
# Helper functions

@torch.no_grad()
def estimate_loss():
    """Estimate mean loss over train/val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """Cosine LR schedule with warm‑up."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def measure_perplexity(split='val', batch_size=1):
    """Strict perplexity evaluation on the full eval set."""
    assert split in ['train', 'val']
    model.eval()
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    data = torch.from_numpy(np.array(data, dtype=np.int64))
    total_tokens = (len(data) // block_size) * block_size
    data = data[:total_tokens].view(-1, block_size)
    nlls = []
    for i in range(0, data.size(0), batch_size):
        xb = data[i : i + batch_size, :-1].to(device)
        yb = data[i : i + batch_size, 1:].to(device)
        with ctx:
            logits, _ = model(xb, yb)
            logits = logits[:, :-1, :].contiguous()
            targets = yb[:, :-1]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='sum',
        )
        nlls.append(loss)
    total_nll = torch.stack(nlls).sum()
    total_tok = data.size(0) * (block_size - 1)
    ppl = torch.exp(total_nll / total_tok)
    model.train()
    return ppl.item()

# ----------------------------------------------------------------------------
# Logging (optional W&B)
wandb=None

if wandb_log and  eval_only == False:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log="all", log_freq=100)
    
# ----------------------------------------------------------------------------
# Training loop

x, y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0


ppl = []
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()# GradScaler for mixed precision
        ppl_val = measure_perplexity(split='val', batch_size=8)
        ppl.append(ppl_val)
        print(f"Strict perplexity over full val.bin: {ppl_val:.4f}")
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log and  eval_only == False:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "val/ppl": ppl_val,
                    "mfu": running_mfu * 100,
                }
            )
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if eval_only:
        ppl_val = measure_perplexity(split='val', batch_size=8)
        print(f"Strict perplexity over full val.bin: {ppl_val:.4f}")
        break

    # Forward‑backward update with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        x, y = get_batch('train')  # prefetch next batch
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    dt = time.time() - t0
    t0 = time.time()
    if iter_num % log_interval == 0:
        loss_val = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        if wandb_log and  eval_only == False:
            wandb.log({"step_loss": loss_val}, step=iter_num)
        print(
            f"iter {iter_num}: loss {loss_val:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )

    iter_num += 1
    local_iter_num += 1

    # Terminate
    if iter_num > max_iters:
        break
        
if wandb_log and  eval_only == False:
    wandb.finish()