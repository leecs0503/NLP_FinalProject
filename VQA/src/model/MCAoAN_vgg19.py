# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch
import torchvision.transforms as transforms

class MLP(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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

    def forward(self, v, k, q, masks):
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
        atted = self.att(v, k, q, masks).transpose(1, 2)
        atted = atted.contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        # print(atted.shape)
        atted = self.linear_merge(atted) #.squeeze(1)
        # print(atted.shape)
        

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

    def forward(self, v, k, q, masks):
        V = self.multi_head(v, k, q, masks)
        I = self.linear_QI(q) + self.linear_VI(V)
        G = torch.sigmoid(self.linear_QG(q) + self.linear_VG(V))
        # return V
        return I * G


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size=2048, dropout = 0.1):
        super(FFN, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ------------------------
# ---- Self Attention ----
# ------------------------

class SAoA(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(SAoA, self).__init__()

        self.aoa = AoA(hidden_size,dropout=dropout)
        self.ffn = FFN(hidden_size,dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, masks):
        x = self.norm1(x + self.dropout1(
            self.aoa(x, x, x, masks)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class GAoA(nn.Module):
    def __init__(self, hidden_size, dropout = 0.1):
        super(GAoA, self).__init__()

        self.aoa1 = AoA(hidden_size)
        self.aoa2 = AoA(hidden_size)
        self.ffn = FFN(hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, x_masks, y, y_masks):

        x = self.norm1(x + self.dropout1(
            self.aoa1(x, x, x, x_masks)
        ))

        x = self.norm2(x + self.dropout2(
            self.aoa2(y, y, x, y_masks)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# --- MAoAC Layers Cascaded by Encoder-Decoder ---
# ------------------------------------------------

class MCAoA(nn.Module):
    def __init__(self, hidden_size, layer = 6):
        super(MCAoA, self).__init__()

        self.enc_list = nn.ModuleList([SAoA(hidden_size) for _ in range(layer)])
        self.dec_list = nn.ModuleList([GAoA(hidden_size) for _ in range(layer)])

    def forward(self, x, x_masks, y, y_masks):
        # Get hidden vector

        x_masks = x_masks.unsqueeze(1).unsqueeze(2)
        y_masks = y_masks.unsqueeze(1).unsqueeze(2)

        for enc in self.enc_list:
            x = enc(x, x_masks)

        for dec in self.dec_list:
            y = dec(y, y_masks, x, x_masks)

        return x, y


from collections import OrderedDict
import copy
from typing import Dict, List
from torch import nn
import torchvision.models as models
from torch import linalg as LA
import torch
import math
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImagePreChannel(nn.Module):
    def __init__(self):
        """
        Args:
        """
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.roi_heads.nms_thresh=0.7
        vgg19_model = models.vgg19(pretrained=True)
        self.out_features = vgg19_model.classifier[-1].in_features
        vgg19_model.classifier = nn.Sequential(*list(vgg19_model.classifier.children())[:-1])
        self.vgg19_model = vgg19_model
        self.embed_size = 1024
        self.mx_img_len = 100
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            normalize
        ])
    
    def forward(self, image: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=[batch_size, 3, 224, 224])
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        all_result = []
        all_score = []
        
        with torch.no_grad():
            result = self.model(image)
            for idx, obj in enumerate(result):
                # obj's scores is descending
                boxes = obj["boxes"]
                scores_a = obj["scores"]
                tx = []
                # print(boxes.shape[0])
                # 전체 이미지도 넣는다.
                tx.append(
                    F.interpolate(
                        self.transform(image[idx].unsqueeze(0)),
                        size=[224, 224],
                        mode='bicubic',
                        align_corners=False,
                    ).squeeze(0)
                )
                scores = [1.]

                for nn, (box,score) in enumerate(zip(boxes,scores_a)):
                    if nn >= self.mx_img_len - 1:
                        break
                    x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                    if x1 == x2 or y1 == y2:
                        print("!!")
                        continue
                    v = image[idx][0:3, y1:y2, x1:x2].unsqueeze(0)
                    v = F.interpolate(v, size=[224, 224], mode='bicubic', align_corners=False)
                    tx.append(self.transform(v.squeeze(0)))
                    scores.append(score)
                scores = torch.tensor(scores).to(device)
                img_tensor = torch.stack(tx)
                
                embed_features = self.vgg19_model(img_tensor) # |sub_image|, out_feature
                if embed_features.shape[0] < self.mx_img_len:
                    scores = torch.cat([
                        scores,
                        torch.zeros(self.mx_img_len - scores.shape[0]).to(device),
                    ])
                    embed_features = torch.cat([
                        embed_features,
                        torch.zeros((self.mx_img_len - embed_features.shape[0], self.out_features)).to(device),
                    ])
                all_result.append(embed_features)
                all_score.append(scores)
        all_result =  torch.stack(all_result)
        all_score = torch.stack(all_score)
        # print(all_result.shape)
        # print(all_mask.shape)
        return all_result, all_score

class ImageChannel(nn.Module):
    def __init__(self, embed_size: int, in_features: int):
        """
        Args:
            embed_size(int): 이미지 채널의 out_features
        """
        super().__init__()
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image_features = self.fc(features)  # (batch_size, num_features, embed_size)
        l2_norm = LA.norm(image_features, ord=2, dim=2, keepdim=True)
        image_features = image_features.div(l2_norm)  # (batch_size, num_features, embed_size)

        return image_features


class TextChannel(nn.Module):
    def __init__(
        self,
        qst_vocab_size: int,
        word_embed_size: int = 300,
        hidden_size: int = 512,
        num_layers: int = 1,
    ):
        """
        Args:
            pretrained_name(str): transformer의 pretrained weight load를 위한 name
            embedding_size(int): word embedding size 
            embed_size(int): output embed size
        """
        super().__init__()
        self.embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=qst_vocab_size, embedding_dim=word_embed_size
        )
        self.lstm = nn.LSTM(
            input_size=word_embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
        )

    # fmt: off
    def forward(self, question):
        """
        Args:
            question_token(dict) : tokenizer를 사용해 얻은 question token
        Return:
            torch.Tensor (shape=[batch_size, embed_size])
        """
        
        embeddings = self.embedding_layer(question)  # [batch_size, max_qst_len, word_embed_size]
        # embeddings = torch.tanh(embeddings)
        qst_features, _ = self.lstm(embeddings)    # [batch_size, max_qst_len, hidden_size]

        return qst_features
    # fmt: on


class MCAoAN(nn.Module):
    def __init__(
        self,
        ans_vocab_size: int,
        qst_vocab_size: int,
        dropout_rate: float,
        embedding_size:int = 300,
        embed_size: int = 1024,
        layer: int = 6,
        multi_modal_dropout: float = 0.2,
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
        self.ImagePreChannel = ImagePreChannel()
        self.image_channel = ImageChannel(embed_size=embed_size, in_features=self.ImagePreChannel.out_features)
        self.text_channel = TextChannel(
            qst_vocab_size=qst_vocab_size,
            word_embed_size=embedding_size,
            hidden_size=embed_size,
        )
        self.mcaoa = MCAoA(hidden_size=embed_size, layer=layer)
        self.mlp1 = MLP(embed_size, dropout_rate)
        self.mlp2 = MLP(embed_size, dropout_rate)
        self.dropout1 = nn.Dropout(multi_modal_dropout)
        self.dropout2 = nn.Dropout(multi_modal_dropout)
        self.fc1 = nn.Linear(2 * embed_size, 2 * embed_size)
        self.fc2 = nn.Linear(2 * embed_size, embed_size)
        self.fc3 = nn.Linear(embed_size, 2)
        self.fc4 = nn.Linear(embed_size, ans_vocab_size)
        self.fc5 = nn.Linear(embed_size, 4)

        self.layernorm = LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(0.5)
        self.fc11 = nn.Linear(embed_size, ans_vocab_size)
        self.fc22 = nn.Linear(ans_vocab_size, ans_vocab_size)

    # fmt: off
    def forward(
        self,
        image_features: torch.Tensor,
        image_masks: torch.Tensor,
        question_embedding: torch.Tensor,
    ):
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=[batch_size, 3, 224, 224])
            question(torch.Tensor): batch 크기만큼 들어있는 질문의 텐서 (shape=[batch_size, max_qst_len])
        Return:
            torch.Tensor (shape = [batch_size, ans_vocab_size])
        """
        img_feature = self.image_channel(image_features)                    # [batch_size, num_features, embed_size]
        qst_masks = question_embedding == 0
        image_masks = image_masks.type(torch.bool)
        qst_feature = self.text_channel(question_embedding)                  # [batch_size, max_qst_len, embed_size]
        qst_feature, img_feature = self.mcaoa(qst_feature, qst_masks, img_feature, image_masks)
        
        qst_attended_feature = self.mlp1(qst_feature)                       # [batch_size, max_qst_len, 1]
        qst_attended_feature.masked_fill_(qst_masks.unsqueeze(2), -1e9)
        qst_attended_feature = F.softmax(qst_attended_feature, dim=1)
        qst_att = qst_attended_feature
        qst_attended_feature = torch.sum(qst_attended_feature * qst_feature, dim=1) # [batch_size, embed_size]
        img_attended_feature = self.mlp2(img_feature)                               # [batch_size, num_features]
        img_attended_feature.masked_fill_(image_masks.unsqueeze(2), -1e9)
        img_attended_feature = F.softmax(img_attended_feature, dim=1)
        img_att = img_attended_feature
        img_attended_feature = torch.sum(img_attended_feature * img_feature, dim=1) # [batch_size, embed_size]

        fused_feature = torch.stack([qst_attended_feature, img_attended_feature], dim=1) # [batch_size, 2, embed_size]
        fused_weight = self.fc1(torch.cat((qst_attended_feature, img_attended_feature), dim=1))
        fused_weight = self.dropout1(fused_weight)
        fused_weight = self.fc2(fused_weight)
        fused_weight = self.dropout2(fused_weight)
        fused_weight = self.fc3(fused_weight)
        fused_weight = F.softmax(fused_weight, dim=1)                   # [batch_size, 2]
        fused_weight = fused_weight.unsqueeze(2)                        # [batch_size, 2, 1]
        combined_feature = torch.sum(fused_weight * fused_feature, dim=1)   # [batch_size,  embed_size]

        # aoa
        # combined_feature = qst_attended_feature + img_attended_feature

        # 임시 코드
        # combined_feature = torch.mul(qst_attended_feature, img_attended_feature)     # [batch_size, embed_size]
        # # combined_feature = torch.mul(qst_attended_feature, img_attended_feature)     # [batch_size, embed_size]
        # combined_feature = torch.tanh(combined_feature)             # [batch_size, embed_size]
        # combined_feature = self.dropout(combined_feature)          # [batch_size, embed_size]
        # combined_feature = self.fc11(combined_feature)              # [batch_size, ans_vocab_size]
        # combined_feature = torch.tanh(combined_feature)             # [batch_size, ans_vocab_size]
        # combined_feature = self.dropout(combined_feature)          # [batch_size, ans_vocab_size]
        # combined_feature = self.fc22(combined_feature)              # [batch_size, ans_vocab_size]
        # return combined_feature, None

        # aoa
        combined_feature = self.layernorm(combined_feature)
        vqa_feature = self.fc4(combined_feature)

        vqa_feature = torch.sigmoid(vqa_feature)              # [batch_size, ans_vocab_size]

        # vg
        # vg_feature = self.fc5(combined_feature)               # [batch_size, 4]
        # vg_feature = vg_feature.sigmoid() * 224
        return vqa_feature, None, qst_att, img_att
    # fmt: on
    def get_name(self):
        return 'MCAoAN_vgg19'
    def get_params(self):
        return self.parameters()
        # (
        #     list(self.image_channel.fc.parameters())
        #     + list(self.text_channel.parameters())
        #     + list(self.mcaoa.parameters())
        #     + list(self.mlp1.parameters())
        #     + list(self.mlp2.parameters())
        #     + list(self.fc1.parameters())
        #     + list(self.fc2.parameters())
        #     + list(self.fc3.parameters())
        #     + list(self.fc4.parameters())
        #     + list(self.fc5.parameters())
        # )
