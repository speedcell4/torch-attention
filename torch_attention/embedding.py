import torch
from torch import nn


class PositionEmbedding(nn.Embedding):
    def __init__(self, max_sentence_length: int, in_features: int) -> None:
        super().__init__(max_sentence_length, in_features)

    def reset_parameters(self) -> None:
        sentence, features = self.weight.size()
        pos = torch.arange(0, sentence).float()
        ixs = torch.arange(0, features).float()
        ixs = 1. / torch.pow(10000., torch.floor(ixs / 2) * 2 / features)
        with torch.no_grad():
            self.weight.data = pos.view(-1, 1) @ ixs.view(1, -1)
            self.weight.data[..., 0::2].sin_()
            self.weight.data[..., 1::2].cos_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batches, dim1, dim2 = inputs.size()
        return self.weight[:dim1, :dim2].expand(*batches, -1, -1)
