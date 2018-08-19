import torch
from torch import nn

from torch_attention import Attention


class DotProduct(Attention):
    def __init__(self) -> None:
        super(DotProduct, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)
