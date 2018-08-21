from hypothesis import given

from tests import *
from torch_attention import BiLinearAttention, DotProduct, Facets, MultiHead


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    in_features1=NORMAL_FEATURES,
    in_features2=NORMAL_FEATURES,
    bias=BIAS,
)
def test_facets(batches, channel1, channel2, in_features1, in_features2, bias):
    attention = Facets(in_features1=in_features1, bias=bias)

    Q = torch.rand(*batches, channel1, in_features1)
    K = torch.rand(*batches, channel2, in_features1)
    V = torch.rand(*batches, channel2, in_features2)
    A = torch.rand(*batches, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, in_features2)
    assert attention(Q, K, V).size() == (*batches, channel1, in_features2)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    in_features1=NORMAL_FEATURES,
    in_features2=NORMAL_FEATURES,
)
def test_dot_product(batches, channel1, channel2, in_features1, in_features2):
    attention = DotProduct()

    Q = torch.rand(*batches, channel1, in_features1)
    K = torch.rand(*batches, channel2, in_features1)
    V = torch.rand(*batches, channel2, in_features2)
    A = torch.rand(*batches, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, in_features2)
    assert attention(Q, K, V).size() == (*batches, channel1, in_features2)


@given(
    batches=BATCHES,
    channel1=CHANNEL,
    channel2=CHANNEL,
    in_features1=NORMAL_FEATURES,
    in_features2=NORMAL_FEATURES,
    num_heads=NUM_HEADS,
    model_features=TINY_FEATURES,
)
def test_multi_head(batches, channel1, channel2, in_features1, in_features2, num_heads, model_features):
    out_features = num_heads * model_features
    attention = MultiHead(
        in_features1=in_features1, in_features2=in_features2,
        num_heads=num_heads, out_features=out_features,
    )

    Q = torch.rand(*batches, channel1, in_features1)
    K = torch.rand(*batches, channel2, in_features1)
    V = torch.rand(*batches, channel2, in_features2)
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
    in_features1=NORMAL_FEATURES,
    in_features2=NORMAL_FEATURES,
    in_features3=NORMAL_FEATURES,
    hidden_size=st.one_of(st.one_of(), NORMAL_FEATURES),
)
def test_bilinear(batches, channel1, channel2, in_features1, in_features2, in_features3, hidden_size):
    attention = BiLinearAttention(in_features1, in_features2, hidden_size)
    Q = torch.rand(*batches, channel1, in_features1)
    K = torch.rand(*batches, channel2, in_features2)
    V = torch.rand(*batches, channel2, in_features3)
    A = torch.rand(*batches, channel1, channel2)

    attention = attention.to(device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    A = A.to(device)

    assert attention.attend(Q, K).size() == A.size()
    assert attention.interact(A, V).size() == (*batches, channel1, in_features3)
    assert attention(Q, K, V).size() == (*batches, channel1, in_features3)
