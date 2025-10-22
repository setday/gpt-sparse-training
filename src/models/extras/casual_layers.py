import torch
import torch.nn as nn
from torch.nn import functional as F

from activations import ReLU2


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
def make_activation_func(name):
    activation_func_map = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "relu^2": ReLU2,
    }
    return activation_func_map.get(name, nn.GELU)()

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., bias=True, activation="gelu"):
        super().__init__()

        self.c_fc       = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = make_activation_func(activation)
        self.c_proj     = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            make_activation_func("gelu"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
