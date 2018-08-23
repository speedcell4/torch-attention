from typing import Tuple
from itertools import chain

import torch
from torch import nn
from torch.nn import init
from more_itertools import interleave

from torch_attention import MultiHeadAttention


class FeedForwardLayer(nn.Sequential):
    def __init__(self, in_features: int, out_features: int = None, hidden_features: int = None, bias: bool = False,
                 window_sizes: Tuple[int, ...] = (1, 5, 1), negative_slope: float = 0., inplace: bool = True) -> None:

        if out_features is None:
            out_features = in_features
        if hidden_features is None:
            hidden_features = max(in_features, out_features)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = len(window_sizes)

        in_dims = chain([in_features], [hidden_features for _ in window_sizes[:-1]])
        out_dims = chain([hidden_features for _ in window_sizes[:-1]], [out_features])

        super(FeedForwardLayer, self).__init__(*interleave(
            [nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
             for _ in window_sizes],
            [nn.Conv1d(in_channels=in_dims, out_channels=out_dims, stride=1,
                       kernel_size=window_size, padding=window_size // 2, bias=bias)
             for in_dims, out_dims, window_size in zip(in_dims, out_dims, window_sizes)],
        ))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self[1::2]:
            init.kaiming_uniform_(layer.weight)
            if getattr(layer, 'bias', None) is not None:
                init.constant_(layer.bias, 0.)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_features: int, num_heads: int, head_features: int = None,
                 window_sizes: Tuple[int, ...] = (1, 5, 1), dropout: float = 0.1, bias: bool = False) -> None:
        super(TransformerEncoderBlock, self).__init__()
        assert in_features % num_heads == 0
        assert all(window_size > 0 and window_size % 2 == 1 for window_size in window_sizes)

        self.out_features = in_features

        self.attention = MultiHeadAttention(
            num_heads=num_heads, head_features=head_features, out_features=in_features,
            q_features=in_features, k_features=in_features, v_features=in_features,
        )
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.layer_norm1 = nn.LayerNorm(in_features)

        self.feed_forward = FeedForwardLayer(
            in_features=in_features, window_sizes=window_sizes, bias=bias,
        )
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.layer_norm2 = nn.LayerNorm(in_features)

    def reset_parameters(self) -> None:
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()

    def forward(self, x: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        y = self.attention(x, x, x, mask)
        z = self.layer_norm1(self.dropout1(y) + x)
        w = self.feed_forward(z.transpose(-2, -1)).transpose(-2, -1)
        return self.layer_norm2(self.dropout2(w) + z)


class TransformerEncoder(nn.Sequential):
    def __init__(self, num_layers: int, in_features: int, num_heads: int, head_features: int = None,
                 window_sizes: Tuple[int, ...] = (1, 5, 1), dropout: float = 0.1, bias: bool = False) -> None:
        super(TransformerEncoder, self).__init__(*[
            TransformerEncoderBlock(
                in_features=in_features, num_heads=num_heads, head_features=head_features,
                dropout=dropout, window_sizes=window_sizes, bias=bias,
            ) for _ in range(num_layers)
        ])

    def reset_parameters(self) -> None:
        for layer in self:
            layer.reset_parameters()


if __name__ == '__main__':
    net = TransformerEncoder(num_layers=3, in_features=512, num_heads=8)
    print(net)
