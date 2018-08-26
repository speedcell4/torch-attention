from abc import ABCMeta

import torch
from torch import nn

from torch_attention.utils import masked_fill


class Attention(nn.Module, metaclass=ABCMeta):
    def __init__(self, q_features: int, k_features: int, v_features: int, out_features: int) -> None:
        super(Attention, self).__init__()

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

    def attend(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def interact(self, A: torch.Tensor, V: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        if mask is not None:
            A = masked_fill(A, mask=mask, filling_value=-float('inf'))
        return self.softmax(A) @ V

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        return self.interact(self.attend(Q, K), V, mask=mask)


from torch_attention.attentions import DotProductAttention, BiLinearAttention, FacetsAttention
from torch_attention.multi_head import MultiHeadAttention
from torch_attention.embedding import PositionEmbedding

__all__ = [
    'masked_fill',
    'Attention',
    'DotProductAttention', 'BiLinearAttention', 'FacetsAttention',
    'MultiHeadAttention',
    'PositionEmbedding',
]
