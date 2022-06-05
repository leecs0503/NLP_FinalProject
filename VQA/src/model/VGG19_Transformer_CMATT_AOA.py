# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2




# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, hidden_size, multi_head = 8, dropout = 0.1):
        super(MHAtt, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_head = hidden_size // multi_head
        self.multi_head = multi_head
        
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)
        atted = self.att(v, k, q).transpose(1, 2)
        atted = atted.contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted).squeeze(1)

        return atted

    def att(self, value, key, query, mask = None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class AoA(nn.Module):
    def __init__(self, hidden_size, multi_head = 8, dropout = 0.1):
        super(AoA, self).__init__()
        self.hidden_size = hidden_size
        self.multi_head = MHAtt(hidden_size, multi_head, dropout)
        self.linear_VG = nn.Linear(hidden_size, hidden_size)
        self.linear_VI = nn.Linear(hidden_size, hidden_size)
        self.linear_QG = nn.Linear(hidden_size, hidden_size)
        self.linear_QI = nn.Linear(hidden_size, hidden_size)

    def forward(self, v, k, q):
        V = self.multi_head(v, k, q)
        I = self.linear_QI(q) + self.linear_VI(V)
        G = nn.functional.sigmoid(self.linear_QG(q) + self.linear_VG(V))
        return torch.mul(I, G)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size=2048, dropout = 0.1):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(SA, self).__init__()

        self.aoa = AoA(hidden_size,dropout=dropout)
        self.ffn = FFN(hidden_size,dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(
            self.aoa(x, x, x)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, hidden_size, dropout = 0.1):
        super(SGA, self).__init__()

        self.aoa1 = AoA(hidden_size)
        self.aoa2 = AoA(hidden_size)
        self.ffn = FFN(hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y):
        x = self.norm1(x + self.dropout1(
            self.aoa1(x, x, x)
        ))

        x = self.norm2(x + self.dropout2(
            self.aoa2(y, y, x)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, hidden_size, layer = 6):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(hidden_size) for _ in range(layer)])
        self.dec_list = nn.ModuleList([SGA(hidden_size) for _ in range(layer)])

    def forward(self, x, y):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x)

        for dec in self.dec_list:
            y = dec(y, x)

        return x, y


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
            pretrained_name(str): transformer의 pretrained weight load를 위한 name
            embedding_size(int): word embedding size 
            embed_size(int): output embed size
        """
        super().__init__()
        self.sbert = AutoModel.from_pretrained(pretrained_name)
        self.fc = nn.Linear(embedding_size, embed_size)

    # fmt: off
    def forward(self, question_token):
        """
        Args:
            question_token(dict) : tokenizer를 사용해 얻은 question token
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        with torch.no_grad():
            question_embedding = self.sbert(
                input_ids=question_token["input_ids"].squeeze(1).to(device),
                token_type_ids=question_token["token_type_ids"].squeeze(1).to(device),
                attention_mask=question_token["attention_mask"].squeeze(1).to(device),
            )
        
        question_embedding = mean_pooling(question_embedding, question_token['attention_mask'].squeeze(1).to(device))
        qst_features = self.fc(question_embedding)                             # [batch_size, embed_size]

        return qst_features
    # fmt: on


class Transformer_VQA_CMATT_AOA(nn.Module):
    def __init__(
        self,
        ans_vocab_size: int,
        dropout_rate: float,
        pretrained_name:str='sentence-transformers/bert-base-nli-mean-tokens',
        embedding_size:int = 768,
        embed_size: int = 1024,
        layer: int = 6
    ):
        """
        Args:
            ans_vocab_size(int): answer vocab의 크기 (output tensor의 크기)
            dropout_rate(int): dropout시 적용할 하이퍼 파라메터
            pretrained_name(str): transformer의 pretrained weight load를 위한 name
            embedding_size(int): word embedding size 
            embed_size(int): output embed size

        Return:
            torch.Tensor (shape=[batch_size, ans_vocab_size])
        """
        super().__init__()
        self.image_channel = ImageChannel(embed_size=embed_size)
        self.text_channel = TextChannel(
            pretrained_name=pretrained_name,
            embedding_size=embedding_size,
            embed_size=embed_size,
        )
        self.cross_modal_attention = MCA_ED(hidden_size=embed_size,layer=layer)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        self.fc3 = nn.Linear(embed_size, 4)

    # fmt: off
    def forward(
        self,
        image: torch.Tensor,
        question_embedding: torch.Tensor,
    ):
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=[batch_size, 3, 224, 224])
            question(torch.Tensor): batch 크기만큼 들어있는 질문의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape = [batch_size, ans_vocab_size])
        """
        img_feature = self.image_channel(image)                    # [batch_size, embed_size]
        qst_feature = self.text_channel(question_embedding)                  # [batch_size, embed_size]
        qst_feature, img_feature = self.cross_modal_attention(qst_feature, img_feature)
        combined_feature = torch.mul(img_feature, qst_feature)     # [batch_size, embed_size]
        combined_feature = torch.tanh(combined_feature)             # [batch_size, embed_size]
        combined_feature = self.dropout(combined_feature)          # [batch_size, embed_size]
        vqa_feature = self.fc1(combined_feature)              # [batch_size, ans_vocab_size]
        vqa_feature = torch.tanh(vqa_feature)             # [batch_size, ans_vocab_size]
        vqa_feature = self.dropout(vqa_feature)          # [batch_size, ans_vocab_size]

        vqa_feature = self.fc2(vqa_feature)              # [batch_size, ans_vocab_size]
        vg_feature = self.fc3(combined_feature)               # [batch_size, 4]
        vg_feature = vg_feature.sigmoid() * 244
        return vqa_feature, vg_feature
    # fmt: on
    def get_name(self):
        return 'VGG19_Transformer_cross_modal_attention(AoA)_with_multijoint_learning'
    def get_params(self):
        return (
            list(self.image_channel.fc.parameters())
            + list(self.text_channel.parameters())
            + list(self.cross_modal_attention.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.fc3.parameters())
        )
