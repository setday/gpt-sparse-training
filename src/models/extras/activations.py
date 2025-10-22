import torch.nn as nn


class ReLU2(nn.ReLU):
    def forward(self, input):
        relu = super().forward(input)
        return relu * relu
