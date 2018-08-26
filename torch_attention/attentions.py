import torch
from torch import nn
from torch.nn import init

from torch_attention import Attention


class DotProductAttention(Attention):
    def __init__(self, q_k_features: int, v_features: int, temperature: float = None) -> None:
        super(DotProductAttention, self).__init__(
            q_features=q_k_features, k_features=q_k_features,
            v_features=v_features, out_features=v_features,
        )
        self.softmax = nn.Softmax(dim=-1)
        if temperature is None:
            temperature = 1.0 / (q_k_features ** 0.5)
        self.temperature = temperature

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return Q @ K.transpose(-2, -1) * self.temperature


class BiLinearAttention(Attention):
    def __init__(self, q_features: int, k_features: int, v_features: int, bias: bool = False) -> None:
        super(BiLinearAttention, self).__init__(
            q_features=q_features, k_features=k_features,
            v_features=v_features, out_features=v_features,
        )

        self.weight = nn.Parameter(torch.Tensor(q_features, k_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(size=()))
        self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'bias'):
            init.constant_(self.bias, 0.)
        init.kaiming_uniform_(self.weight, nonlinearity='sigmoid')

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        a = torch.einsum('...qx,xy,...ky->...qk', (Q, self.weight, K))
        return a + self.bias if hasattr(self, 'bias') else a


class FacetsAttention(Attention):
    def __init__(self, q_k_features: int, v_features: int, bias: bool = False) -> None:
        super(FacetsAttention, self).__init__(
            q_features=q_k_features, k_features=q_k_features,
            v_features=v_features, out_features=v_features,
        )

        self.fc = nn.Linear(q_k_features * 4, 1, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        *batches, channel1, in_features = Q.size()
        *batches, channel2, in_features = K.size()
        sizes = (*batches, channel1, channel2, in_features)
        Q = Q.view(*batches, channel1, 1, in_features).expand(*sizes)
        K = K.view(*batches, 1, channel2, in_features).expand(*sizes)
        return self.fc(torch.cat([
            Q, K, (Q - K).abs(), Q * K
        ], dim=-1)).squeeze(-1)
