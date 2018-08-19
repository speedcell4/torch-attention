import torch
from torch import nn

from torch_attention import Attention


class Facets(Attention):
    def __init__(self, in_features1: int, bias: bool = False) -> None:
        super(Facets, self).__init__()
        self.in_features1 = in_features1

        self.fc = nn.Linear(in_features1 * 4, 1, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        assert Q.ndimension() == K.ndimension(), f'{Q.ndimension()} != {K.ndimension()}'

        *batches, channel1, in_features1 = Q.size()
        *batches, channel2, in_features1 = K.size()
        Q = Q.view(*batches, channel1, 1, in_features1).expand(*batches, channel1, channel2, in_features1)
        K = K.view(*batches, 1, channel2, in_features1).expand(*batches, channel1, channel2, in_features1)
        return self.fc(torch.cat([Q, K, (Q - K).abs(), Q * K], dim=-1)).squeeze(-1)
