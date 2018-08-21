from hypothesis import given

from tests import *
from torch_attention import BiLinearAttention, DotProduct, Facets, MultiHead


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    in_features=NORMAL_FEATURES,
    bias=BIAS,
    device=DEVICE,
)
def test_facets(batches, channel1, channel2, in_features, bias, device):
    attention = Facets(in_features=in_features, bias=bias)

    Q = torch.rand(*batches, channel1, in_features)
    K = torch.rand(*batches, channel2, in_features)
    V = torch.rand(*batches, channel2, in_features)
    A = torch.rand(*batches, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, in_features)
    assert attention(Q, K, V).size() == (*batches, channel1, in_features)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    k_features=NORMAL_FEATURES,
    v_features=NORMAL_FEATURES,
    device=DEVICE,
)
def test_dot_product(batches, channel1, channel2, k_features, v_features, device):
    attention = DotProduct()

    Q = torch.rand(*batches, channel1, k_features)
    K = torch.rand(*batches, channel2, k_features)
    V = torch.rand(*batches, channel2, v_features)
    A = torch.rand(*batches, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, v_features)
    assert attention(Q, K, V).size() == (*batches, channel1, v_features)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    k_features=NORMAL_FEATURES,
    v_features=NORMAL_FEATURES,
    num_heads=NUM_HEADS,
    model_features=TINY_FEATURES,
    device=DEVICE,
)
def test_multi_head(batches, channel1, channel2, k_features, v_features, num_heads, model_features, device):
    out_features = num_heads * model_features
    attention = MultiHead(
        k_features=k_features, v_features=v_features,
        num_heads=num_heads, out_features=out_features,
    )

    Q = torch.rand(*batches, channel1, k_features)
    K = torch.rand(*batches, channel2, k_features)
    V = torch.rand(*batches, channel2, v_features)
    A = torch.rand(*batches, num_heads, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, out_features)
    assert attention(Q, K, V).size() == (*batches, channel1, out_features)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    q_features=NORMAL_FEATURES,
    k_features=NORMAL_FEATURES,
    v_features=NORMAL_FEATURES,
    hidden_size=st.one_of(st.none(), NORMAL_FEATURES),
    device=DEVICE,
)
def test_bilinear(batches, channel1, channel2, q_features, k_features, v_features, hidden_size, device):
    attention = BiLinearAttention(q_features, k_features, hidden_size)
    Q = torch.rand(*batches, channel1, q_features)
    K = torch.rand(*batches, channel2, k_features)
    V = torch.rand(*batches, channel2, v_features)
    A = torch.rand(*batches, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, v_features)
    assert attention(Q, K, V).size() == (*batches, channel1, v_features)
