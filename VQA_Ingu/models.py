import torch
import torch.nn as nn
import torchvision.models as models


class ImageModel(nn.modules):
    def __init__(self, embed_size):
        super(self, ImageModel).__init__()
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
        self.model = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(in_features, embed_size)


    def forward(self, image):
        with torch.no_grad():
            image_feature = self.model(image)
        norm = torch.linalg.norm(A=image_feature, dim=1, keepdim=True).detach()
        image_feature = image_feature.div(norm)
        image_feature = self.fc(image_feature)
        return image_feature


class QuestionModel(nn.modules):
    def __init__(self, question_vocabulary_size, word_embed_size, embed_size, num_layer, hidden_layer_size):
        super(self, QuestionModel).__init__()
        self.input_embedding = nn.Embedding(question_vocabulary_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_layer_size, num_layer)
        self.fc = nn.Linear(2 * num_layer * hidden_layer_size, embed_size)

    def forward(self, question):
        question_vector = self.input_embedding(question)
        question_vector = self.tanh(question_vector)
        _, (hidden, cell) = self.lstm(question_vector)
        output_feature = torch.cat((hidden, cell))
        output_feature = self.tanh(output_feature)
        output_feature = self.fc(output_feature)
        return output_feature


class VQAModel(nn.modules):
    def __init__(
        self,
        embed_size,
        question_vocabulary_size,
        answer_vocabulary_size,
        word_embed_size,
        num_layer,
        hidden_layer_size
    ):
        super(self, VQAModel).__init__()
        self.image_encoder = ImageModel(embed_size)
        self.question_encoder = QuestionModel(
            question_vocabulary_size,
            word_embed_size,
            embed_size,
            num_layer,
            hidden_layer_size
        )
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, answer_vocabulary_size)
        self.fc2 = nn.Linear(answer_vocabulary_size, answer_vocabulary_size)

    def forward(self, image, question):
        image_feature = self.image_encoder(image)
        question_feature = self.question_encoder(question)
        combine_feature = torch.mul(image_feature, question_feature)
        combine_feature = self.tanh(combine_feature)
        combine_feature = self.dropout(combine_feature)
        combine_feature = self.fc1(combine_feature)
        combine_feature = self.tanh(combine_feature)
        combine_feature = self.dropout(combine_feature)
        combine_feature = self.fc2(combine_feature)
        combine_feature = self.softmax(combine_feature)

        return combine_feature
