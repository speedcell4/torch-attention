import torch
from torch import nn
from torch.nn import init

from torch_attention import Attention, masked_fill


class MultiHead(Attention):
    def __init__(self, k_features: int, v_features: int, num_heads: int, out_features: int) -> None:
        super(MultiHead, self).__init__()

        assert num_heads >= 1
        assert out_features % num_heads == 0

        self.num_heads = num_heads
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.Q = nn.Parameter(torch.Tensor(self.k_features, self.out_features))
        self.K = nn.Parameter(torch.Tensor(self.k_features, self.out_features))
        self.V = nn.Parameter(torch.Tensor(self.v_features, self.out_features))
        self.W = nn.Parameter(torch.Tensor(num_heads, self.out_features // num_heads, self.out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            init.kaiming_uniform_(self.Q)
            init.kaiming_uniform_(self.K)
            init.kaiming_uniform_(self.V)
            init.kaiming_uniform_(self.W)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        assert Q.ndimension() == K.ndimension(), f'{Q.ndimension()} != {K.ndimension()}'

        Q = (Q @ self.Q).view(*Q.size()[:-1], self.num_heads, -1)
        K = (K @ self.K).view(*K.size()[:-1], self.num_heads, -1)
        return torch.einsum('...qhx,...khx->...hqk', (Q, K)) / (K.size(-1) ** 0.5)

    def interact(self, A: torch.Tensor, V: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        assert A.ndimension() == V.ndimension() + 1, f'{A.ndimension()} != {V.ndimension()} + 1'

        V = (V @ self.V).view(*V.size()[:-1], self.num_heads, -1)
        if mask is not None:
            A = masked_fill(A, mask, filling_value=-float('inf'))
        R = torch.einsum('...hqk,...khx->...qhx', (A, V))
        return torch.einsum('...qhx,hxy->...qy', (R, self.W))
