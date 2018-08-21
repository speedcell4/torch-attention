import torch
from torch import nn
from torch.nn import init

from torch_attention import Attention


class BiLinearAttention(Attention):
    def __init__(self, q_features: int, k_features: int, hidden_size: int = None):
        super(BiLinearAttention, self).__init__()
        if hidden_size is None:
            hidden_size = max(q_features, k_features)
        self.Q = nn.Parameter(torch.Tensor(q_features, hidden_size))
        self.K = nn.Parameter(torch.Tensor(k_features, hidden_size))
        self.activation = nn.ReLU(inplace=True)

        self.W = nn.Parameter(torch.Tensor(hidden_size, 1, hidden_size))
        self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.Q)
        init.kaiming_uniform_(self.K)
        init.kaiming_uniform_(self.W)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q = self.activation(Q @ self.Q)
        K = self.activation(K @ self.K)
        return torch.einsum('...qx,xzy,...ky->...qkz', (Q, self.W, K)).squeeze(-1)
