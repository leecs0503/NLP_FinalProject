import torch
from util.text_helper import VocabDict
from models import VQAModel
from typing import List
from data_loader import VQA_Dataset, VQA_Input_Data
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model_data():
    model_path = "./models/model-epoch-30.ckpt"
    qst_vocab_path = "./datasets/vocab_questions.txt"
    ans_vocab_path = "./datasets/vocab_answers.txt"
    train_data_path = './datasets/test.npy'
    max_qst_length = 30
    max_num_ans = 10

    qst_vocab_dict = VocabDict(qst_vocab_path)
    ans_vocab_dict = VocabDict(ans_vocab_path)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    test_vqa_data: List(VQA_Input_Data) = np.array(
        [VQA_Input_Data(x) for x in np.load(train_data_path, allow_pickle=True)]
    )

    qst_vocab_size = qst_vocab_dict.vocab_size
    ans_vocab_size = ans_vocab_dict.vocab_size
    ans_unk_idx = ans_vocab_dict.unk2idx

    model = VQAModel(
        image_model_name= "vgg19",
        embed_size = 1024,
        word_embed_size = 300,
        num_layers = 2,
        hidden_size = 512,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
    )
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.to(device)
    model.eval()


    for idx, vqa_data in enumerate(test_vqa_data):
        if idx > 100:
            break;
        vqa_data.image_path
        image_rgb = Image.open(vqa_data.image_path).convert("RGB")
        image = torch.stack([transform(image_rgb)]).to(device)
        save_path = f'./testsets/{idx}'
        os.makedirs(save_path, exist_ok = True)
        image_rgb.save(f"{save_path}/image.png", "png")

        quest_idx_list = np.array(
            [qst_vocab_dict.word2idx("<pad>")] * max_qst_length
        )
        quest_idx_list[: len(vqa_data.question_tokens)] = [
            qst_vocab_dict.word2idx(w) for w in vqa_data.question_tokens
        ]
        
        question = torch.LongTensor([quest_idx_list]).to(device)
        res = model(image, question)
        _, pred_exp1 = torch.max(res, 1)  # [batch_size]
        _, pred_exp2 = torch.max(res, 1)  # [batch_size]
        answer = ans_vocab_dict.word_list[pred_exp1.cpu()[0].item()]
        with open(f'./testsets/{idx}/qna.txt', "w") as f:
            f.write(f"Question: {vqa_data.question_str}\n")
            f.write(f"Answer: {answer}")
        print(f"{idx} done")
        

if __name__ == '__main__':
    test_model_data()