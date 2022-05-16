from torch import nn
import torchvision.models as models
from torch import linalg as LA
import torch
import math


class ImageChannel(nn.Module):
    def __init__(self, embed_size: int):
        """
        Args:
            embed_size(int): 이미지 채널의 out_features
        """
        super().__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        self.model = model
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=[batch_size, 3, 224, 224])
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        with torch.no_grad():
            image_features = self.model(image)
        image_features = self.fc(image_features)  # (batch_size, embed_size)

        l2_norm = LA.norm(image_features, ord=2, dim=1, keepdim=True)
        normalized = image_features.div(l2_norm)  # (batch_size, embed_size)

        return normalized


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_qst_length: int = 30):
        """
        Args:
            hidden_size(int):
            max_qst_length(int):
        Return
            x: torch.Tensor
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_qst_length, hidden_size)
        position = torch.arange(0, max_qst_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def generate_mask(src: torch.Tensor, pad_token: int):
    """각 batch에 대해 token이 <pad>이면 1, 아니면 0 인 텐서를 반환하는 함수
    Args:
        src(torch.T)
        pad_token:
    Return: torch.Tensor [batch_size, qst_len, qst_len]
    """
    batch_size = src.shape[0]
    qst_len = src.shape[-1]

    pad_attn_mask = torch.where(
        src == pad_token, torch.ones(src.shape), torch.zeros(src.shape)
    ).to(torch.bool)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, qst_len, qst_len)
    return pad_attn_mask


class TextChannel(nn.Module):
    def __init__(
        self,
        qst_vocab_size: int,
        pad_token: int,
        hidden_size: int = 512,
        num_head: int = 8,
        dim_feedforward: int = 2048,
        max_qst_length: int = 30,
        num_encode_layers: int = 6,
        embed_size: int = 1024,
    ):
        """
        Args:
            qst_vocab_size(int): qustion vocab의 크기
            hidden_size(int): transformer의 output tensor 크기 (default 512)
            num_head(int): transformer의
            dim_feedforward(int): transformer의 feedforward의 은닉층의 차원
            num_encoder_layers(int): transformer encoder을 몇 층 쌓을 건지
            max_qst_length(int): 가장 긴 질문의 길이
            embed_size(int): 최종 output layer의 크기
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_token = pad_token
        self.num_head = num_head
        self.embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=qst_vocab_size, embedding_dim=hidden_size
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
        )
        self.positional_encoding = PositionalEncoding(hidden_size, max_qst_length)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encode_layers,
        )
        self.fc = nn.Linear(hidden_size * num_encode_layers, embed_size)

    # fmt: off
    def forward(self, question: torch.Tensor):
        """
        Args:
            question(torch.Tensor): batch 크기만큼 들어있는 질의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        embeddings = self.embedding_layer(question) * math.sqrt(self.hidden_size)  # [batch_size, max_qst_len, hidden_size]
        embeddings = self.positional_encoding(embeddings)                          # [batch_size, max_qst_len, hidden_size]
        mask = generate_mask(question, self.pad_token)                             # [batch_size, hidden_size, hidden_size]
        mask = mask.repeat(self.num_head, 1, 1)                                    # [batch_size * num_head, hidden_size, hidden_size]

        embeddings = embeddings.transpose(0, 1)                                 # [max_qst_len, batch_size, hidden_size]
        output = self.transformer(embeddings, mask=mask)                        # [num_encoder, batch_size, hidden_size]
        output = output.transpose(0, 1)                                         # [batch_size, num_encoder, hidden_size]
        output = output.reshape(output.shape[0], -1)                            # [batch_size, num_encoder * hidden_size]
        qst_features = self.fc(output)                                          # [batch_size, embed_size]

        return qst_features
    # fmt: on


class Transformer_VQA(nn.Module):
    def __init__(
        self,
        ans_vocab_size: int,
        dropout_rate: float,
        qst_vocab_size: int,
        pad_token: int,
        embed_size: int = 1024,
        hidden_size: int = 512,
        num_head: int = 8,
        dim_feedforward: int = 2048,
        num_encode_layers: int = 6,
        max_qst_length: int = 30,
    ):
        """
        Args:
            ans_vocab_size(int): answer vocab의 크기 (output tensor의 크기),
            dropout_rate(int): dropout시 적용할 하이퍼 파라메터,
            qst_vocab_size(int): qustion vocab의 크기
            pad_token(int): <pad>의 token 값
            embed_size(int): 이미지 체널과 텍스트 체널에 embed 크기 (default: 1024)
            hidden_size(int): transformer의 output tensor 크기 (default: 512)
            num_head(int): transformer의 attention head 개수 (deafult: 8)
            dim_feedforward(int): transformer의 feedforward의 은닉층의 차원 (default:2048)
            num_encoder_layers(int): transformer encoder을 몇 층 쌓을 건지 (default: 6)
            max_qst_length(int): 가장 긴 질문의 길이 (default:30)
        Return:
            torch.Tensor (shape=[batch_size, ans_vocab_size])
        """
        super().__init__()
        self.image_channel = ImageChannel(embed_size=embed_size)
        self.text_channel = TextChannel(
            qst_vocab_size=qst_vocab_size,
            pad_token=pad_token,
            hidden_size=hidden_size,
            num_head=num_head,
            dim_feedforward=dim_feedforward,
            num_encode_layers=num_encode_layers,
            max_qst_length=max_qst_length,
            embed_size=embed_size,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    # fmt: off
    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
    ):
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=[batch_size, 3, 224, 224])
            question(torch.Tensor): batch 크기만큼 들어있는 질의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape = [batch_size, ans_vocab_size])
        """
        img_feature = self.image_channel(image)                    # [batch_size, embed_size]
        qst_feature = self.text_channel(question)                  # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)     # [batch_size, embed_size]
        combined_feature = torch.tanh(combined_feature)             # [batch_size, embed_size]
        combined_feature = self.dropout(combined_feature)          # [batch_size, embed_size]
        combined_feature = self.fc1(combined_feature)              # [batch_size, ans_vocab_size]
        combined_feature = torch.tanh(combined_feature)             # [batch_size, ans_vocab_size]
        combined_feature = self.dropout(combined_feature)          # [batch_size, ans_vocab_size]
        combined_feature = self.fc2(combined_feature)              # [batch_size, ans_vocab_size]
        return combined_feature
    # fmt: on

    def get_params(self):
        return (
            list(self.image_channel.fc.parameters())
            + list(self.text_channel.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
        )
