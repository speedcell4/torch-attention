from abc import ABCMeta

import torch
from torch import nn


class Attention(nn.Module, metaclass=ABCMeta):
    def attend(self, Q: torch.Tensor, K: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        raise NotImplementedError

    def interact(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return A @ V

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        return self.interact(self.attend(Q, K, mask=mask), V)
