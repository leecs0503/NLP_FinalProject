from src.model.VGG19_Tansformer import (
    ImageChannel,
    TextChannel,
    Transformer_VQA,
    generate_mask,
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_ImageChannel():
    batch_size, embed_size = 3, 10

    model = ImageChannel(embed_size).to("cpu")
    model.eval()

    image = torch.zeros((batch_size, 3, 244, 244))
    res = model(image)
    assert res.shape == torch.Size([batch_size, embed_size])


def test_PositionalEncoding():
    # todo: implement test code
    return


def test_generate_mask():
    src = torch.tensor([[3, 1, 2, 0, 0, 0]], dtype=torch.int32).to(device)
    expected = torch.tensor(
        [
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ]
        ],
        dtype=torch.int32,
    ).to(device)
    result = generate_mask(src, 0)

    assert expected.size() == result.size()
    assert torch.all(torch.eq(expected, result))
    assert result.dtype == torch.bool


def test_TextChannel():
    embed_size = 10

    model = TextChannel(
        qst_vocab_size=10,
        pad_token=0,
        hidden_size=8,
        num_head=8,
        dim_feedforward=2048,
        max_qst_length=6,
        num_encode_layers=6,
        embed_size=embed_size,
    ).to(device)
    model.eval()

    # batch_size = 2
    question = torch.stack(
        [
            torch.tensor([3, 1, 2, 0, 0, 0], dtype=torch.int32).to(device),
            torch.tensor([5, 2, 3, 4, 1, 0], dtype=torch.int32).to(device),
        ]
    )

    res = model(question)
    assert res.shape == torch.Size([2, embed_size])


def test_Transformer_VQA():
    batch_size, embed_size, ans_vocab_size = 2, 10, 5
    model = Transformer_VQA(
        ans_vocab_size=ans_vocab_size,
        dropout_rate=0.5,
        qst_vocab_size=10,
        pad_token=0,
        embed_size=embed_size,
        hidden_size=8,
        max_qst_length=6,
    ).to(device)
    model.eval()
    image = torch.zeros((batch_size, 3, 244, 244)).to(device)
    question = torch.stack(
        [
            torch.tensor([3, 1, 2, 0, 0, 0], dtype=torch.int32).to(device),
            torch.tensor([5, 2, 3, 4, 1, 0], dtype=torch.int32).to(device),
        ]
    )
    res = model(question=question, image=image)
    assert res.shape == torch.Size([batch_size, ans_vocab_size])
