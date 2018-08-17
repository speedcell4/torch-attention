from typing import NoReturn

import torch
from torch import nn
from torch.nn import init

from torch_attention import Attention, DotProduct


class MultiHead(Attention):
    def __init__(self, in_features1: int, in_features2: int, num_heads: int, out_features: int):
        super(MultiHead, self).__init__()

        assert num_heads >= 1
        assert out_features % num_heads == 0

        self.num_heads = num_heads
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features

        self.Q = nn.Parameter(torch.Tensor(self.in_features1, self.out_features))
        self.K = nn.Parameter(torch.Tensor(self.in_features1, self.out_features))
        self.V = nn.Parameter(torch.Tensor(self.in_features2, self.out_features))
        self.W = nn.Parameter(torch.Tensor(self.out_features, self.out_features))

        self.dot_product = DotProduct()

        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        with torch.no_grad():
            init.orthogonal_(self.Q)
            init.orthogonal_(self.K)
            init.orthogonal_(self.V)
            init.kaiming_uniform_(self.W)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        Q = (Q @ self.Q).view(*Q.size()[:-1], self.num_heads, -1)
        K = (K @ self.K).view(*K.size()[:-1], self.num_heads, -1)
        V = (V @ self.V).view(*V.size()[:-1], self.num_heads, -1)
        A = self.dot_product(
            Q=Q.transpose(-2, -3),
            K=K.transpose(-2, -3),
            V=V.transpose(-2, -3),
            mask=mask,
        ).transpose(-2, -3)
        return A.contiguous().view(*A.size()[:-2], -1) @ self.W
