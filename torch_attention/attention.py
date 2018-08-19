from abc import ABCMeta

import torch
from torch import nn

from torch_attention import masked_fill


class Attention(nn.Module, metaclass=ABCMeta):
    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def interact(self, A: torch.Tensor, V: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        assert A.ndimension() == V.ndimension(), f'{A.ndimension()} != {V.ndimension()}'

        if mask is not None:
            A = masked_fill(A, mask=mask, filling_value=-float('inf'))
        return self.softmax(A) @ V

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        return self.interact(self.attend(Q, K), V, mask=mask)
