from torch import nn
import torchvision.models as models
from torch import linalg as LA
import torch


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


class TextChannel(nn.Module):
    def __init__(
        self,
        qst_vocab_size: int,
        word_embed_size: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        embed_size: int = 1024,
    ):
        """
        Args:
            qst_vocab_size(int): qustion vocab의 크기
            word_embed_size(int): word embedding할 크기 (default 300)
            hidden_size(int): LSTM의 hidden layer 크기 (default 512)
            num_layers(int): stack된 LSTM의 개수 (default 2)
        """
        super().__init__()
        self.embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=qst_vocab_size, embedding_dim=word_embed_size
        )
        self.lstm = nn.LSTM(
            input_size=word_embed_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)

    # fmt: off
    def forward(self, question: torch.Tensor):
        """
        Args:
            question(torch.Tensor): batch 크기만큼 들어있는 질의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        embeddings = self.embedding_layer(question)  # [batch_size, max_qst_len, word_embed_size]
        embeddings = torch.tanh(embeddings)
        embeddings = embeddings.transpose(0, 1)      # [max_qst_len, batch_size, word_embed_size]
        _, (hidden, cell) = self.lstm(embeddings)    # [num_layer, batch_size, word_embed_size]
        qst_features = torch.cat((hidden, cell), 2)  # [num_layer, batch_size, 2*hidden_size]
        qst_features = qst_features.transpose(0, 1)  # [batch_size, num_layer, 2*hidden_size]
        qst_features = qst_features.reshape(qst_features.size()[0], -1)  # [batch_size, 2*num_layer*hidden_size]
        qst_features = torch.tanh(qst_features)
        qst_features = self.fc(qst_features)  # [batch_size, embed_size]

        return qst_features
    # fmt: on


class LSTM_VQA(nn.Module):
    def __init__(
        self,
        embed_size: int,
        qst_vocab_size: int,
        word_embed_size: int,
        num_layers: int,
        hidden_size: int,
        ans_vocab_size: int,
        dropout_rate: float,
    ):
        """
        Args:
            embed_size(int): 이미지 체널과 텍스트 체널에 ,
            qst_vocab_size(int): qustion vocab의 크기
            word_embed_size(int): word embedding할 크기 (default 300)
            num_layers(int): stack된 LSTM의 개수 (default 2)
            hidden_size(int): LSTM의 hidden layer 크기 (default 512)
            ans_vocab_size(int): answer vocab의 크기 (output tensor의 크기),
            dropout_rate(int): dropout시 적용할 하이퍼 파라메터,
        Return:
            torch.Tensor (shape=[batch_size, ans_vocab_size])
        """
        super().__init__()
        self.image_channel = ImageChannel(embed_size=embed_size)
        self.text_channel = TextChannel(
            qst_vocab_size=qst_vocab_size,
            word_embed_size=word_embed_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
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
