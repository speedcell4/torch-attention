from hypothesis import given

from torch_attention import PositionEmbedding
from tests import *


@given(
    batches=BATCHES,
    channel=CHANNEL,
    addition_embedding=CHANNEL,
    in_features=NORMAL_FEATURES,
)
def test_position_embedding(batches, channel, in_features, addition_embedding):
    num_embeddings = channel + addition_embedding
    layer = PositionEmbedding(num_embeddings, in_features)
    x = torch.rand(*batches, channel, in_features)
    y = layer(x)

    assert y.size() == (*batches, channel, in_features)
