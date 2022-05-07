from torch import nn
import torchvision.models as models
import torch


class ImageChannel(nn.Module):
    def __init__(self, embed_size: int):
        """
        Args:
            embed_size(int): 이미지 채널의 out_features
        """
        super().__init__()
        model = models.vgg19(pretrained=True)
        in_features = models.classifier[-1].in_feature
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

        l2_norm = image_features.norm(p=2, dim=1, keepdim=True)  # (batch_size)
        normalized = image_features.div(l2_norm)  # (batch_size, embed_size)

        return normalized


class TextChannel(nn.Module):
    def __init__(
        self,
        ques_vocab_size: int,
        word_embed_size: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        embed_size: int = 1024,
    ):
        """
        Args:
            ques_vocab_size(int): question vocab의 크기
            word_embed_size(int): word embedding할 크기 (default 300)
            hidden_size(int): LSTM의 hidden layer 크기 (default 512)
            num_layers(int): stack된 LSTM의 개수 (default 2)
        """
        super().__init__()
        self.embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=ques_vocab_size, embedding_dim=word_embed_size
        )
        self.lstm = nn.LSTM(
            input_size=word_embed_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.fc = nn.Linear(2 * num_layers * word_embed_size, embed_size)

    def forward(self, question: torch.Tensor):
        """
        Args:
            question(torch.Tensor): batch 크기만큼 들어있는 질의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        embeddings = self.embedding_layer(
            question
        )  # [batch_size, max_qst_len, word_embed_size]
        embeddings = torch.tanh(embeddings)
        embeddings = embeddings.transpose(
            0, 1
        )  # [max_qst_len, batch_size, word_embed_size]
        _, (hidden, cell) = self.lstm(
            embeddings
        )  # [num_layer, batch_size, word_embed_size]
        qst_features = torch.cat(
            (hidden, cell), 2
        )  # [num_layer, batch_size, 2 * word_embed_size]
        qst_features = qst_features.transpose(
            0, 1
        )  # [batch_size, num_layer, 2 * word_embed_size]
        qst_features = qst_features.reshape(
            qst_features.size()[0], -1
        )  # [batch_size, 2 * num_layer * word_embed_size]

        qst_features = torch.tanh(qst_features)
        qst_features = self.fc(qst_features)  # [batch_size, embed_size]

        return qst_features


class LSTM_VQA(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
