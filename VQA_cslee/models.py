import torch
import torch.nn as nn
import torchvision.models as models
import abc


class Model(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        pass

    @abc.abstractmethod
    def forward(self, **kwargs):
        pass


class ImgEncoder(Model):
    def __init__(self, model_name, embed_size):
        super().__init__()
        self.model = None
        if model_name == "vgg19":
            model = models.vgg19(pretrained=True)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            self.model = model
            self.fc = nn.Linear(in_features, embed_size)
            pass
        else:
            raise Exception("Unexpected Model Name")
        pass

    def forward(self, **kwargs):
        image = kwargs["image"]

        with torch.no_grad():
            image_feature = self.model(image)
        image_feature = self.fc(image_feature)

        l2_norm = image_feature.norm(p=2, dim=1, keepdim=True).detach()
        image_feature = image_feature.div(l2_norm)

        return image_feature


class QstEncoder(Model):
    def __init__(
        self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size
    ):
        super().__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)

    def forward(self, **kwargs):
        question = kwargs["question"]

        qst_vec = self.word2vec(
            question
        )  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(
            0, 1
        )  # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(
            qst_vec
        )  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat(
            (hidden, cell), 2
        )  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(
            0, 1
        )  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(
            qst_feature.size()[0], -1
        )  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature


class VQAModel(Model):
    def __init__(
        self,
        embed_size,
        qst_vocab_size,
        ans_vocab_size,
        word_embed_size,
        num_layers,
        hidden_size,
    ):
        super().__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(
            qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size
        )
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, **kwargs):
        image = kwargs["image"]
        question = kwargs["question"]

        img_feature = self.img_encoder(image)  # [batch_size, embed_size]
        qst_feature = self.qst_encoder(question)  # [batch_size, embed_size]

        combined_feature = torch.mul(
            img_feature, qst_feature
        )  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(
            combined_feature
        )  # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(
            combined_feature
        )  # [batch_size, ans_vocab_size=1000]

        return combined_feature
