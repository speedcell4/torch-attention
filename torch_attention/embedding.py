import torch
from torch import nn

from torch_attention.init import position_


class PositionEmbedding(nn.Module):
    def __init__(self, max_dim1: int, max_dim2: int) -> None:
        super(PositionEmbedding, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(max_dim1, max_dim2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        position_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batches, dim1, dim2 = inputs.size()
        return self.weight[:dim1, :dim2].expand(*batches, -1, -1)
