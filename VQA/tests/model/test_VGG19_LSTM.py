from src.model.VGG19_LSTM import ImageChannel, TextChannel, LSTM_VQA
import torch


def test_ImageChannel():
    batch_size, embed_size = 3, 10

    model = ImageChannel(embed_size).to("cpu")
    model.eval()

    image = torch.zeros((batch_size, 3, 244, 244))
    res = model(image)
    assert res.shape == torch.Size([batch_size, embed_size])


def test_TextChannel():
    embed_size = 10

    model = TextChannel(
        qst_vocab_size=10,
        word_embed_size=20,
        hidden_size=8,
        num_layers=2,
        embed_size=embed_size,
    ).to("cpu")
    model.eval()

    # batch_size = 2
    question = torch.stack(
        [
            torch.tensor([3, 1, 2, 0, 0, 0], dtype=torch.int32, device="cpu"),
            torch.tensor([5, 2, 3, 4, 1, 0], dtype=torch.int32, device="cpu"),
        ]
    )

    res = model(question)
    assert res.shape == torch.Size([2, embed_size])


def test_LSTM_VQA():
    batch_size, embed_size, ans_vocab_size = 2, 10, 5
    model = LSTM_VQA(
        embed_size=embed_size,
        qst_vocab_size=10,
        word_embed_size=20,
        num_layers=2,
        hidden_size=8,
        ans_vocab_size=ans_vocab_size,
        dropout_rate=0.5,
    ).to("cpu")
    model.eval()
    image = torch.zeros((batch_size, 3, 244, 244))
    question = torch.stack(
        [
            torch.tensor([3, 1, 2, 0, 0, 0], dtype=torch.int32, device="cpu"),
            torch.tensor([5, 2, 3, 4, 1, 0], dtype=torch.int32, device="cpu"),
        ]
    )
    res = model(question=question, image=image)
    assert res.shape == torch.Size([batch_size, ans_vocab_size])
