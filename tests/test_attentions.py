import torch
from hypothesis import given, strategies as st

from torch_attention.dot_product import DotProduct
from torch_attention.multi_head import MultiHead

hypo = dict(
    batches=st.lists(st.integers(1, 10), min_size=1, max_size=5),
    channel1=st.integers(1, 10),
    channel2=st.integers(1, 10),
    in_features1=st.integers(20, 100),
    in_features2=st.integers(20, 100),
)


@given(**hypo)
def test_dot_product(batches, channel1, channel2, in_features1, in_features2):
    dot_product = DotProduct()

    Q = torch.rand(*batches, channel1, in_features1)
    K = torch.rand(*batches, channel2, in_features1)
    V = torch.rand(*batches, channel2, in_features2)

    assert dot_product(Q, K, V).size() == (*batches, channel1, in_features2)


@given(
    **hypo,
    num_heads=st.integers(1, 10),
    model_features=st.integers(5, 20),
)
def test_multi_head(batches, channel1, channel2, in_features1, in_features2, num_heads, model_features):
    out_features = num_heads * model_features
    dot_product = MultiHead(
        in_features1=in_features1, in_features2=in_features2,
        num_heads=num_heads, out_features=out_features,
    )

    Q = torch.rand(*batches, channel1, in_features1)
    K = torch.rand(*batches, channel2, in_features1)
    V = torch.rand(*batches, channel2, in_features2)

    assert dot_product(Q, K, V).size() == (*batches, channel1, out_features)