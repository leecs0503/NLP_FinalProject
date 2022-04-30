import torch
import torch.nn as nn
import torchvision.models as models


class ImageModel(nn.Module):
    def __init__(self, embed_size):
        super(ImageModel, self).__init__()
        model = models.resnet101(pretrained=True)
        in_features = model.fc.out_features
        # in_features = model.classifier[-1].in_features
        # model.classifier = nn.Sequential(*(list(model.classifier.children())[:-1]))
        # model = nn.Sequential(*(list(model.children())[:-1]))
        self.model = model
        self.fc = nn.Linear(in_features, embed_size)


    def forward(self, image):
        with torch.no_grad():
            image_feature = self.model(image)
        image_feature = image_feature.squeeze()
        image_feature = self.fc(image_feature)
        norm = torch.linalg.vector_norm(image_feature, ord=2, dim=1, keepdim=True).detach()
        image_feature = image_feature.div(norm)
        return image_feature


class QuestionModel(nn.Module):
    def __init__(self, question_vocabulary_size, word_embed_size, embed_size, num_layer, hidden_layer_size):
        super(QuestionModel, self).__init__()
        self.input_embedding = nn.Embedding(question_vocabulary_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_layer_size, num_layer)
        self.fc = nn.Linear(2 * num_layer * hidden_layer_size, embed_size)

    def forward(self, question):
        question_vector = self.input_embedding(question)
        question_vector = self.tanh(question_vector)
        question_vector = question_vector.transpose(0, 1)
        _, (hidden, cell) = self.lstm(question_vector)
        output_feature = torch.cat((hidden, cell), 2)
        output_feature = output_feature.transpose(0, 1)
        output_feature = output_feature.reshape(output_feature.size()[0], -1)
        output_feature = self.tanh(output_feature)
        output_feature = self.fc(output_feature)
        return output_feature


class VQAModel(nn.Module):
    def __init__(
        self,
        embed_size,
        question_vocabulary_size,
        answer_vocabulary_size,
        word_embed_size,
        num_layers,
        hidden_layer_size
    ):
        super(VQAModel, self).__init__()
        self.image_encoder = ImageModel(embed_size)
        self.question_encoder = QuestionModel(
            question_vocabulary_size,
            word_embed_size,
            embed_size,
            num_layers,
            hidden_layer_size
        )
        self.softmax = nn.Softmax(dim=1)
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
