import torch


def position_(tensor: torch.Tensor) -> None:
    sentence, features = tensor.size()
    pos = torch.arange(0, sentence).float()
    ixs = torch.arange(0, features).float()
    ixs = 1. / torch.pow(10000., torch.floor(ixs / 2) * 2 / features)
    with torch.no_grad():
        tensor.data = pos.view(-1, 1) @ ixs.view(1, -1)
        tensor.data[..., 0::2].sin_()
        tensor.data[..., 1::2].cos_()
