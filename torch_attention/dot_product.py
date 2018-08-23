import torch
from torch import nn

from torch_attention import Attention


class DotProductAttention(Attention):
    def __init__(self) -> None:
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        assert Q.ndimension() == K.ndimension(), f'{Q.ndimension()} != {K.ndimension()}'

        return Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)
