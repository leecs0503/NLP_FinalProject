from src.model.VGG19_Tansformer import (
    ImageChannel,
    TextChannel,
    Transformer_VQA,
)
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', model_max_length=30)

# TODO: 현재 구현체에 맞게 테스트코드 변경

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        embed_size=embed_size,
    ).to(device)
    model.eval()

    # batch_size = 2
    question = [
        "3 1 2",
        "5 2 3 4 1",
    ]
    question_token = tokenizer(question, padding='max_length', truncation=True, return_tensors='pt', max_length=6)

    print(question_token)

    res = model(question_token)
    assert res.shape == torch.Size([2, embed_size])


def test_Transformer_VQA():
    batch_size, embed_size, ans_vocab_size = 2, 10, 5
    
    model = Transformer_VQA(
        ans_vocab_size=ans_vocab_size,
        dropout_rate=0.5,
        embed_size=embed_size,
    ).to(device)
    model.eval()
    image = torch.zeros((batch_size, 3, 244, 244)).to(device)
    question = [
        "3 1 2",
        "5 2 3 4 1",
    ]
    question_token = tokenizer(question, padding='max_length', truncation=True, return_tensors='pt', max_length=6)
    res = model(question_embedding=question_token, image=image)
    assert res.shape == torch.Size([batch_size, ans_vocab_size])
