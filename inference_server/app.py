import kserve
import argparse
from typing import Dict
import os
import sys
import torch
module_path = os.path.abspath(os.path.join('./.download_modules'))
sys.path.append(module_path)

from VQA.src.model.MCAoAN_vgg19 import MCAoAN
from VQA.src.model.VGG19_LSTM import LSTM_VQA
from VQA.src.model.VGG19_Transformer import Transformer_VQA
from VQA.src.utils.vocab_dict import VocabDict
from VQA.src.utils import text_helper
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import base64
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def plt_to_base64_str(plt):
    my_stringIObytes = BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    return my_base64_jpgData.decode()

class VQA_Model(kserve.Model):
    def __init__(self, name: str, args):
        super().__init__(name)
        self.name = name
        self.model_load(
            model_dir=args.model_dir,
            vocab_dir=args.vocab_dir,
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize
        ])
        self.max_sub_img_num = args.max_sub_img_num

    def model_load(
        self,
        model_dir: str,
        vocab_dir: str,
    ):
        self.qst_vocab = VocabDict(os.path.join(vocab_dir, 'vocab_questions.txt'))
        self.ans_vocab = VocabDict(os.path.join(vocab_dir, 'vocab_answers.txt'))
        self.qst_vocab_size = self.qst_vocab.vocab_size
        self.ans_vocab_size = self.ans_vocab.vocab_size
        
        # model load

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load faster rcnn model

        self.rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.rcnn_model.roi_heads.nms_thresh = 0.7
        self.rcnn_model.eval().to(self.device)

        # load mcaoan model
        self.mcaoan_model = MCAoAN(
            ans_vocab_size= self.ans_vocab_size,
            qst_vocab_size= self.qst_vocab_size,
            dropout_rate= 0.5,
            embed_size=64,
        )
        self.mcaoan_model.load_state_dict(torch.load(args.model_dir)["model_state_dict"])
        self.mcaoan_model.eval().to(self.device)

        # load LSTM model
        self.lstm_model = LSTM_VQA(

        )

        # load SBERT model


    def tokenize_question(self, question):
        quest_idx_list = np.array(
            [self.qst_vocab.word2idx("<pad>")] * 30
        )
        question_tokens = text_helper.tokenize(question)
        quest_idx_list[: len(question_tokens)] = [
            self.qst_vocab.word2idx(w) for w in question_tokens
        ]
        question_tokens = torch.tensor(quest_idx_list, dtype=torch.int64, device=self.device).unsqueeze(0)
        return question_tokens, quest_idx_list

    def preprocess(self, inputs: Dict) -> Dict:
        base64_string = inputs["base64image"]
        image = Image.open(BytesIO(base64.b64decode(base64_string)))
        
        question = inputs["question"]

        return image, question

    def predict(self, obj) -> Dict:
        (image_rgb, question) = obj
        image_tensor = self.transform(image_rgb)[0: 3].unsqueeze(0).to(self.device)
        image_feat, image_score = self.mcaoan_model.ImagePreChannel(image_tensor)
        
        score_thr = 0.2
        image_feat = image_feat[:self.max_sub_img_num, :]
        image_score = image_score[:self.max_sub_img_num, :]
        image_mask = image_score < score_thr

        question_tokens, quest_idx_list = self.tokenize_question(question)

        res_rcnn = self.rcnn_model(image_tensor)
        print(image_feat.shape)
        print(image_mask.shape)
        res_MCAoAN = self.mcaoan_model(image_feat, image_mask, question_tokens)
        
        vqa_ans = res_MCAoAN[0]
        qst_att = res_MCAoAN[2]
        img_att = res_MCAoAN[3]
        _, pred_exp = torch.max(vqa_ans, 1)
        ans = self.ans_vocab.idx2word(pred_exp)

        question_data = [
            {
                "word": self.qst_vocab.idx2word(quest_idx),
                "att": qst_att[0][idx].item(),
            }
            for idx, quest_idx in enumerate(quest_idx_list)
            if quest_idx > 0
        ]


        imgs = [
            {
                "box": torch.Tensor([0, 0, image_tensor.shape[3], image_tensor.shape[2]]),
                "att": img_att[0][0]
            } 
        ] + [
            {
                "box": res_rcnn[0]["boxes"][i],
                "att": img_att[0][i + 1]
            } for i in range(0, self.max_sub_img_num - 1) if image_mask[0][i + 1] == False
        ]
        imgs = sorted(imgs, key=lambda x : x["att"].item(), reverse=True)

        num_dict = [0, 0, 0, 0, 0]
        for x in range(0, self.max_sub_img_num):
            num_dict[x%5] += 1
        color_dict = \
            ['red'] * num_dict[0] \
                + ['yellow'] * num_dict[1] \
                + ['green'] * num_dict[2] \
                + ['blue'] * num_dict[3] \
                + ['navy'] * num_dict[4]

        fig,ax = plt.subplots(1)
        ax.imshow(image_rgb)
        ori_img_str = plt_to_base64_str(plt)

        fig,ax = plt.subplots(1)
        ax.imshow(image_rgb)
        for idx, obj in enumerate(imgs):
            box = obj["box"]
            att = obj["att"]
            x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            box_plot = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color_dict[idx], linewidth=1)
            plt.text(x1, y1, f"{idx}: {att.item():.2f}")
            ax.add_patch(box_plot)
        boxed_image_str = plt_to_base64_str(plt)
        important_boxes = []
        for idx, obj in enumerate(imgs):
            box = obj["box"]
            att = obj["att"]
            if att < 1.1 / len(imgs):
                continue
            if idx > 5:
                break
            x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            fig, ax = plt.subplots(1)
            img_pil = Image.fromarray(np.array(image_rgb)[y1:y2,x1:x2,0:3])
            ax.imshow(img_pil)
            print(f"{att.item():.2f}")
            important_boxes.append({
                "att": att.item(),
                "str": plt_to_base64_str(plt),
            })

        return {
            "ori_question": question,
            "ori_image": ori_img_str,
            "question_data": question_data, 
            "boxed_image": boxed_image_str,
            "important_boxes": important_boxes,
            "answer": ans,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directories

    parser.add_argument(
        "--MCAoAN_model_dir",
        type=str,
        default="./models/MCAoAN_vgg19_img10_emb64-epoch-13.ckpt",
        help="directory for model file",
    )
    
    parser.add_argument(
        "--LSTM_model_dir",
        type=str,
        default="./models/VGG19+LSTM-epoch-17.ckpt",
        help="directory for model file",
    )
    
    parser.add_argument(
        "--SBERT_model_dir",
        type=str,
        default="./models/VGG19_Tansformer.py",
        help="directory for model file",
    )
    
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64,
        help="mcaoan model's embedding size",
    )

    parser.add_argument(
        "--max_sub_img_num",
        type=int,
        default=30,
        help="mcaoan model's process image num",
    )
    
    
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default="./vocab",
        help="directory for model file",
    )
    args = parser.parse_args()

    model = VQA_Model("vqa-model", args)
    kserve.ModelServer().start([model])