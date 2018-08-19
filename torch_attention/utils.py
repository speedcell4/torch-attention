import torch


def masked_fill(tensor: torch.Tensor, mask: torch.ByteTensor,
                filling_value: float = -float('inf')) -> torch.Tensor:
    *batch, dim = mask.size()
    mask = mask.view(*batch, *(1,) * (tensor.dim() - mask.dim()), dim)
    return tensor.masked_fill(mask, filling_value)
