from torch import nn
import torchvision.models as models
from torch import linalg as LA
import torch
import math
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class TextChannel(nn.Module):
    def __init__(
        self,
        pretrained_name: str = 'sentence-transformers/bert-base-nli-mean-tokens',
        embedding_size: int = 768,
        embed_size: int = 1024,
    ):
        """
        Args:
        """
        super().__init__()
        self.sbert = AutoModel.from_pretrained(pretrained_name)
        self.fc = nn.Linear(embedding_size, embed_size)

    # fmt: off
    def forward(self, question_token: torch.Tensor):
        """
        Args:
            ~~
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        with torch.no_grad():
            question_embedding = self.sbert(
                input_ids=question_token.input_ids.squeeze(1),
                token_type_ids=question_token.token_type_ids.squeeze(1),
                attention_mask=question_token.attention_mask.squeeze(1),
            )
        
        question_embedding = mean_pooling(question_embedding, question_token['attention_mask'].squeeze(1))
        qst_features = self.fc(question_embedding)                             # [batch_size, embed_size]

        return qst_features
    # fmt: on


class Transformer_VQA(nn.Module):
    def __init__(
        self,
        ans_vocab_size: int,
        dropout_rate: float,
        pretrained_name:str='sentence-transformers/bert-base-nli-mean-tokens',
        embedding_size:int = 768,
        embed_size: int = 1024,
    ):
        """
        Args:
            ~~~
        Return:
            torch.Tensor (shape=[batch_size, ans_vocab_size])
        """
        super().__init__()
        self.image_channel = ImageChannel(embed_size=embed_size)
        self.text_channel = TextChannel(
            embedding_size= 768,
            embed_size=1024,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    # fmt: off
    def forward(
        self,
        image: torch.Tensor,
        question_embedding: torch.Tensor,
    ):
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=[batch_size, 3, 224, 224])
            question(torch.Tensor): batch 크기만큼 들어있는 질의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape = [batch_size, ans_vocab_size])
        """
        img_feature = self.image_channel(image)                    # [batch_size, embed_size]
        qst_feature = self.text_channel(question_embedding)                  # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)     # [batch_size, embed_size]
        combined_feature = torch.tanh(combined_feature)             # [batch_size, embed_size]
        combined_feature = self.dropout(combined_feature)          # [batch_size, embed_size]
        combined_feature = self.fc1(combined_feature)              # [batch_size, ans_vocab_size]
        combined_feature = torch.tanh(combined_feature)             # [batch_size, ans_vocab_size]
        combined_feature = self.dropout(combined_feature)          # [batch_size, ans_vocab_size]
        combined_feature = self.fc2(combined_feature)              # [batch_size, ans_vocab_size]
        return combined_feature
    # fmt: on
    def get_name(self):
        return 'VGG19+Transformer'
    def get_params(self):
        return (
            list(self.image_channel.fc.parameters())
            + list(self.text_channel.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
        )
