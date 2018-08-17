import torch
from torch import nn

from torch_attention import Attention, masked_fill


class DotProduct(Attention):
    def __init__(self):
        super(DotProduct, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        A = Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)
        if mask is not None:
            A = masked_fill(A, mask=mask, filling_value=-float('inf'))
        return self.softmax(A) @ V
