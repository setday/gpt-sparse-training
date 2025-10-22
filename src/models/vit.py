from dataclasses import dataclass
from typing import Literal
import inspect

import torch
from torch import nn
from torch.nn import functional as F

from einops import repeat
from einops.layers.torch import Rearrange

from extras.casual_layers import FeedForward
from extras.transformer_layers import Attention

from sparsify_activations_layer import replace_linears_with_pruner

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Block(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0., bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, bias=bias)
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ln_2 = nn.LayerNorm(dim, bias=bias)
        self.mlp = FeedForward(dim, mlp_dim, dropout = dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

@dataclass
class ViTConfig:
    vocab_size: int = 200 # for TinyImageNet
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    emb_dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    activation_function: Literal["gelu", "relu", "relu^2"] = "gelu" # the non-linear activation function in the MLP block

    # Image-specific
    image_size: int = 64
    patch_size: int = 16
    channels: int = 3

class ViT(nn.Module):
    def __init__(self, config: ViTConfig, sparsity_ratio = 0.0, sparsity_type = None, sparsity_scale = None, mode = "all", custom_slice=None):
        super().__init__()
        self.config = config

        image_height, image_width = pair(config.image_size)
        patch_height, patch_width = pair(config.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.n_embd),
            nn.LayerNorm(config.n_embd),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.n_embd))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.dropout = nn.Dropout(config.emb_dropout)

        self.transformer = Transformer(
            dim=config.n_embd,
            depth=config.n_layer,
            heads=config.n_head,
            dim_head=config.n_embd // config.n_head,
            mlp_dim=config.n_embd * 4,
            dropout=config.dropout
        )

        self.mlp_head = nn.Linear(config.n_embd, config.vocab_size)

        
        self.pruned_layers = replace_linears_with_pruner(
            self,
            sparsity_ratio=sparsity_ratio,
            sparsity_type=sparsity_type,
            sparsity_scale=sparsity_scale,
            mode=mode,
            custom_slice=custom_slice
        )

    def forward(self, img, targets=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.mlp_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the mlp_head on the very last position
            logits = self.mlp_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
