import torch
from torch import nn
from torch.nn import init

from torch_attention import MultiHead


class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_features: int, num_heads: int = 8,
                 dropout: float = 0.1, window_size: int = 5, bias: bool = False) -> None:
        super(TransformerEncoderBlock, self).__init__()
        assert window_size > 0 and window_size % 2 == 1
        assert in_features % num_heads == 0

        self.in_features = in_features
        self.multi_head = MultiHead(
            out_features=in_features, num_heads=num_heads,
            k_features=in_features, v_features=in_features,
        )
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.layer_norm1 = nn.LayerNorm(in_features)

        self.feed_forward = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=in_features, out_channels=in_features, stride=1,
                kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=in_features, out_channels=in_features, stride=1,
                kernel_size=window_size, padding=window_size // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=in_features, out_channels=in_features, stride=1,
                kernel_size=1, padding=0, bias=bias),
        )
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.layer_norm2 = nn.LayerNorm(in_features)

        self.reset_parameters()

    def reset_parameters(self):
        self.multi_head.reset_parameters()
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()
        init.xavier_uniform_(self.feed_forward[1].weight)
        init.xavier_uniform_(self.feed_forward[3].weight)
        init.xavier_uniform_(self.feed_forward[5].weight)

    def forward(self, x: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        y = self.multi_head(x, x, x, mask)
        z = self.layer_norm1(self.dropout1(y) + x)
        w = self.feed_forward(z.transpose(-2, -1)).transpose(-2, -1)
        return self.layer_norm2(self.dropout2(w) + z)


class TransformerEncoder(nn.Sequential):
    def __init__(self, num_layers: int, in_features: int, num_heads: int,
                 dropout: float, window_size: int = 5, bias: bool = False) -> None:
        block = TransformerEncoderBlock(
            in_features=in_features, num_heads=num_heads,
            dropout=dropout, window_size=window_size, bias=bias,
        )
        super(TransformerEncoder, self).__init__(*[
            block for _ in range(num_layers)
        ])
