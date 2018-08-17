from abc import ABCMeta

import torch
from torch import nn


class Attention(nn.Module, metaclass=ABCMeta):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        raise NotImplementedError
