import torch
from torch import nn
from torch.nn import init

from torch_attention import Attention


def multi_head(a: torch.Tensor, num_heads: int) -> torch.Tensor:
    return a.view(*a.size()[:-1], num_heads, -1).transpose_(-2, -3)


class MultiHeadAttention(Attention):
    def __init__(self, q_features: int, k_features: int, v_features: int, out_features: int,
                 attention: Attention, num_heads: int, head_features: int) -> None:
        assert attention.q_features == head_features
        assert attention.k_features == head_features
        assert attention.v_features == head_features
        assert attention.out_features == head_features

        super(MultiHeadAttention, self).__init__(
            q_features=q_features, k_features=k_features,
            v_features=v_features, out_features=out_features,
        )

        self.num_heads = num_heads
        self.head_features = head_features

        self.attention = attention
        self.Q = nn.Parameter(torch.Tensor(q_features, num_heads * head_features))
        self.K = nn.Parameter(torch.Tensor(k_features, num_heads * head_features))
        self.V = nn.Parameter(torch.Tensor(v_features, num_heads * head_features))
        self.W = nn.Parameter(torch.Tensor(num_heads, head_features, out_features))

        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'head={self.num_heads}, head_features={self.head_features}, out_features={self.out_features}, ' \
               f'q_features={self.q_features}, k_features={self.k_features}, v_features={self.v_features}'

    def reset_parameters(self) -> None:
        with torch.no_grad():
            init.kaiming_uniform_(self.Q)
            init.kaiming_uniform_(self.K)
            init.kaiming_uniform_(self.V)
            init.kaiming_uniform_(self.W)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q = multi_head(Q @ self.Q, self.num_heads)
        K = multi_head(K @ self.K, self.num_heads)
        return self.attention.attend(Q, K)

    def interact(self, A: torch.Tensor, V: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        V = multi_head(V @ self.V, self.num_heads)
        R = self.attention.interact(A, V, mask=mask)
        return torch.einsum('...hqx,hxy->...qy', (R, self.W))
