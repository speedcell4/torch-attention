import torch
from hypothesis import strategies as st
from hypothesis import given

from tests import BATCHES, CHANNEL, DEVICE, NUM_HEADS, SMALL_FEATURES, TINY_FEATURES
from torch_attention import BiLinearAttention, DotProductAttention, FacetsAttention, MultiHeadAttention


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    q_features=TINY_FEATURES,
    k_features=TINY_FEATURES,
    v_features=TINY_FEATURES,
    num_heads=NUM_HEADS,
    head_features=SMALL_FEATURES,
    out_features=TINY_FEATURES,
    device=DEVICE,

)
def test_multi_head_with_dot_product(
        batches, channel1, channel2, q_features, k_features, v_features,
        num_heads, head_features, out_features, device):
    head = MultiHeadAttention(
        q_features=q_features, k_features=k_features, v_features=v_features,
        num_heads=num_heads, out_features=out_features, head_features=head_features,
        attention=DotProductAttention(
            q_k_features=head_features, v_features=head_features,
        )
    )

    Q = torch.rand(*batches, channel1, q_features)
    K = torch.rand(*batches, channel2, k_features)
    V = torch.rand(*batches, channel2, v_features)
    A = torch.rand(*batches, num_heads, channel1, channel2)

    head = head.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert head.attend(Q, K).size() == A.size()
    assert head.interact(A, V).size() == (*batches, channel1, out_features)
    assert head(Q, K, V).size() == (*batches, channel1, out_features)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    q_features=TINY_FEATURES,
    k_features=TINY_FEATURES,
    v_features=TINY_FEATURES,
    num_heads=NUM_HEADS,
    head_features=SMALL_FEATURES,
    out_features=TINY_FEATURES,
    device=DEVICE,
    bias=st.booleans(),
)
def test_multi_head_with_bi_linear(
        batches, channel1, channel2, q_features, k_features, v_features,
        num_heads, head_features, out_features, device, bias):
    head = MultiHeadAttention(
        q_features=q_features, k_features=k_features, v_features=v_features,
        num_heads=num_heads, out_features=out_features, head_features=head_features,
        attention=BiLinearAttention(
            q_features=head_features, k_features=head_features, v_features=head_features,
            bias=bias,
        )
    )

    Q = torch.rand(*batches, channel1, q_features)
    K = torch.rand(*batches, channel2, k_features)
    V = torch.rand(*batches, channel2, v_features)
    A = torch.rand(*batches, num_heads, channel1, channel2)

    head = head.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert head.attend(Q, K).size() == A.size()
    assert head.interact(A, V).size() == (*batches, channel1, out_features)
    assert head(Q, K, V).size() == (*batches, channel1, out_features)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    q_features=TINY_FEATURES,
    k_features=TINY_FEATURES,
    v_features=TINY_FEATURES,
    num_heads=NUM_HEADS,
    head_features=SMALL_FEATURES,
    out_features=TINY_FEATURES,
    device=DEVICE,
    bias=st.booleans(),
)
def test_multi_head_with_facets(
        batches, channel1, channel2, q_features, k_features, v_features,
        num_heads, head_features, out_features, device, bias):
    head = MultiHeadAttention(
        q_features=q_features, k_features=k_features, v_features=v_features,
        num_heads=num_heads, out_features=out_features, head_features=head_features,
        attention=FacetsAttention(
            q_k_features=head_features, v_features=head_features,
            bias=bias,
        )
    )

    Q = torch.rand(*batches, channel1, q_features)
    K = torch.rand(*batches, channel2, k_features)
    V = torch.rand(*batches, channel2, v_features)
    A = torch.rand(*batches, num_heads, channel1, channel2)

    head = head.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert head.attend(Q, K).size() == A.size()
    assert head.interact(A, V).size() == (*batches, channel1, out_features)
    assert head(Q, K, V).size() == (*batches, channel1, out_features)
