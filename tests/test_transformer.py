import torch
from hypothesis import given
from hypothesis import strategies as st

from torch_attention import TransformerEncoder
from tests import BATCH, CHANNEL, DEVICE, NUM_HEADS, NUM_LAYERS, TINY_FEATURES


@given(
    batch=BATCH,
    channel=CHANNEL,
    num_layers=NUM_LAYERS,
    model_features=TINY_FEATURES,
    num_heads=NUM_HEADS,
    dropout=st.floats(0., 1.),
    window_size=st.sampled_from([1, 3, 5]),
    bias=st.booleans(),
    device=DEVICE,
)
def test_transformer_encoder(
        batch, channel, num_layers, model_features,
        num_heads, dropout, window_size, bias, device):
    in_features = num_heads * model_features
    encoder = TransformerEncoder(num_layers, in_features, num_heads, dropout, window_size, bias)
    x = torch.rand(batch, channel, in_features)

    encoder = encoder.to(device)
    x = x.to(device)

    assert encoder(x).size() == (batch, channel, in_features)
