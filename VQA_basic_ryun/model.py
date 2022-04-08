from types import CellType
import torch
import torch.nn as nn
import torchvision.models as models

class ImgEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImgEncoder, self).__init__()  # 상속받은 부모클래스(nn.Module)의 속성 사용을 위함
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  # input size
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1]) # 마지막 layer 제외, *는 unpacking?
        
        self.model = model
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image):
        with torch.no_grad():   # inference 시에 속도향상을 위해 autograd를 끔
            img_feature = self.model(image)
        img_feature = self.fc(img_feature)

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach() # p = order of norm, .detach() tensor 기록 추적 중단
        img_feature = img_feature.div(l2_norm)  # l2 norm 적용

        return img_feature

class QstEncoder(nn.Module):
    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)   # 임베딩할 단어의 개수, 임베딩 벡터 차원 입력
        self.tanh = nn.Tanh()  
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)

    def forward(self, question):
        qst_vec = self.word2vec(question)
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)
        _, (hidden, cell) = self.lstm(qst_vec)
        qst_feature = torch.cat((hidden, cell), 2)
        qst_featrue = qst_featrue.transpose(0, 1)
        qst_featrue = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  

        return qst_feature

class VqaModel(nn.Module):
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):
        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

        img_feature = self.img_encoder(img)                     
        qst_feature = self.qst_encoder(qst)                     
        combined_feature = torch.mul(img_feature, qst_feature)  
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)   

        return combined_feature