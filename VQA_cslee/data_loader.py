import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from typing import List, Dict
from PIL import Image
from util.text_helper import VocabDict
import numpy as np


class VQA_Input_Data:
    def __init__(self, vqa_data: Dict):
        self.image_name = vqa_data["image_name"]
        self.image_path = vqa_data["image_path"]
        self.question_id = vqa_data["question_id"]
        self.question_str = vqa_data["question_str"]
        self.question_tokens = vqa_data["question_tokens"]
        self.all_answers = (
            vqa_data["all_answers"] if "all_answers" in vqa_data else None
        )
        self.valid_answers = (
            vqa_data["valid_answers"] if "valid_answers" in vqa_data else None
        )


class VQA_Data:
    def __init__(self, image, question, answer_label=None, answer_multi_choice=None):
        self.image = image
        self.question = question
        self.answer_label = answer_label
        self.answer_multi_choice = answer_multi_choice

    def to_dict(self):
        return {
            "image": self.image,
            "question": self.question,
            "answer_label": self.answer_label,
            "answer_multi_choice": self.answer_multi_choice,
        }


class VQA_Dataset(data.Dataset):
    def __init__(
        self,
        dataset: List[VQA_Input_Data],
        max_qst_length: int,
        max_num_ans: int,
        transform=None,
        question_dict: VocabDict = None,
        answer_dict: VocabDict = None,
    ):
        """VQA의 Dataset"""
        self.dataset = dataset
        self.load_ans = dataset[0].valid_answers is not None
        self.question_dict = question_dict
        self.answer_dict = answer_dict
        self.transform = transform
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans

    def __len__(self):
        """dataset에 들어있는 데이터 개수"""
        return len(self.dataset)

    def __getitem__(self, index: int) -> VQA_Data:
        vqa_data = self.dataset[index]

        # preprocess image
        image_rgb = Image.open(vqa_data.image_path).convert("RGB")
        image = self.transform(image_rgb) if self.transform else image_rgb

        # preprocess question
        quest_idx_list = np.array(
            [self.question_dict.word2idx("<pad>")] * self.max_qst_length
        )
        quest_idx_list[: len(vqa_data.question_tokens)] = [
            self.question_dict.word2idx(w) for w in vqa_data.question_tokens
        ]

        # preprocess answer
        answer_label = None
        answer_multi_choice = None
        if self.load_ans:
            ans2idc = [self.answer_dict.word2idx(w) for w in vqa_data.valid_answers]
            ans2idx = np.random.choice(ans2idc)
            answer_label = ans2idx

            mul2idc = list(
                [-1] * self.max_num_ans
            )  # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[: len(ans2idc)] = ans2idc  # our model should not predict -1
            answer_multi_choice = mul2idc  # for evaluation metric of 'multiple choice'
        return VQA_Data(
            image=image,
            question=quest_idx_list,
            answer_label=answer_label,
            answer_multi_choice=answer_multi_choice,
        ).to_dict()

    def get_vocab_size(self, vocab_type):
        if vocab_type == "question":
            return self.question_dict.vocab_size
        if vocab_type == "answer":
            return self.answer_dict.vocab_size
        raise Exception("invalid params (vocab_type)")


def get_vqa_data_loader(
    train_data_path: str,  #
    valid_data_path: str,  #
    qst_vocab_path: str,  #
    ans_vocab_path: str,  #
    max_qst_length: int,  #
    max_num_ans: int,  #
    batch_size: int,  #
    num_workers: int,  #
):
    """ """
    qst_vocab_dict = VocabDict(qst_vocab_path)
    ans_vocab_dict = VocabDict(ans_vocab_path)

    train_vqa_data = np.array(
        [VQA_Input_Data(x) for x in np.load(train_data_path, allow_pickle=True)]
    )
    valid_vqa_data = np.array(
        [VQA_Input_Data(x) for x in np.load(valid_data_path, allow_pickle=True)]
    )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    vqa_dataset = {
        "train": VQA_Dataset(
            dataset=train_vqa_data,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform,
            question_dict=qst_vocab_dict,
            answer_dict=ans_vocab_dict,
        ),
        "valid": VQA_Dataset(
            dataset=valid_vqa_data,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform,
            question_dict=qst_vocab_dict,
            answer_dict=ans_vocab_dict,
        ),
    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for phase in ["train", "valid"]
    }
    return data_loader
