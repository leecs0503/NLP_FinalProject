from typing import TypedDict, List, Dict, Optional
import torch.utils.data
from .utils.vocab_dict import VocabDict
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# batch_step_size = len(self.data_loader[phase].dataset) / batch_size


class VQA_DataLoader(TypedDict):
    train: torch.utils.data.DataLoader
    valid: torch.utils.data.DataLoader


class VQA_Raw_Data:
    def __init__(self, vqa_data: Dict):
        self.image_name: str = vqa_data["image_name"]
        self.image_path: str = vqa_data["image_path"]
        self.question_id: str = vqa_data["question_id"]
        self.question_str: str = vqa_data["question_str"]
        self.question_tokens: List[str] = vqa_data["question_tokens"]
        self.all_answers: Optional[List[str]] = (
            vqa_data["all_answers"] if "all_answers" in vqa_data else None
        )
        self.valid_answers: Optional[List[str]] = (
            vqa_data["valid_answers"] if "valid_answers" in vqa_data else None
        )


class VQA_Input_Data:
    def __init__(
        self,
        image,
        question,
        answer_label=None,
        answer_multi_choice=None,
    ):
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


class VQA_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: List[VQA_Raw_Data],
        max_qst_length: int,
        max_num_ans: int,
        question_dict: VocabDict,
        answer_dict: Optional[VocabDict] = None,
        transform: Optional[transforms.Compose] = None,
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

    def __getitem__(self, index: int) -> VQA_Input_Data:
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
        answer_label = -1
        answer_multi_choice = []
        if self.load_ans:
            ans2idc = [self.answer_dict.word2idx(w) for w in vqa_data.valid_answers]
            answer_label = np.random.choice(ans2idc)

            mul2idc = list(
                [-1] * self.max_num_ans
            )  # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[: len(ans2idc)] = ans2idc  # our model should not predict -1
            answer_multi_choice = mul2idc  # for evaluation metric of 'multiple choice'
        return VQA_Input_Data(
            image=image,
            question=quest_idx_list,
            answer_label=answer_label,
            answer_multi_choice=answer_multi_choice,
        ).to_dict()

    def get_vocab_size(self, vocab_type: str):
        if vocab_type == "question":
            return self.question_dict.vocab_size
        if vocab_type == "answer":
            return self.answer_dict.vocab_size
        raise Exception("invalid params (vocab_type)")


def load_data_loader(
    data_path: str,
    qst_vocab_dict: VocabDict,
    ans_vocab_dict: VocabDict,
    max_qst_length: int,
    max_num_ans: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    # default normalize value = vgg19 normalizer
    normalize: transforms.Normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
) -> torch.utils.data.DataLoader:
    vqa_data = np.array(
        [VQA_Raw_Data(x) for x in np.load(data_path, allow_pickle=True)]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    vqa_dataset = VQA_Dataset(
        dataset=vqa_data,
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        transform=transform,
        question_dict=qst_vocab_dict,
        answer_dict=ans_vocab_dict,
    )
    return torch.utils.data.DataLoader(
        dataset=vqa_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def load_VQA_DataLoader(
    train_data_path: str,  #
    valid_data_path: str,  #
    qst_vocab_dict: VocabDict,
    ans_vocab_dict: VocabDict,
    max_qst_length: int,
    max_num_ans: int,
    batch_size: int,
    num_workers: int,
    normalize: transforms.Normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
):
    """trainer에서 활용할 DataLoader을 불러오는 메소드
    Args:
        train_data_path: train시      preprocess된 vqa 데이터(train.npy)가 있는 경로
        valid_data_path: validation시 preprocess된 vqa 데이터(valid.npy)가 있는 경로
        qst_vocab_dict:  질문에 대한 VocabDict
        ans_vocab_dict:  정답에 대한 VocabDict
        max_qst_length:  질문의 최대 길이를 몇으로 할 것인지
        max_num_ans:     정답의 최대 길이
        batch_size:      한 배치에 받아드릴 입력의 크기
        num_workers:     데이터 로드시 worker의 개수
        normalize:       pretrain된 이미지 모델을 활용하기 위해 사전 학습된 모델의 이미지 정규화 기법
    Returns:
        torch.Tensor: 모델이 반환한 텐서
    """
    return {
        "train": load_data_loader(
            data_path=train_data_path,
            qst_vocab_dict=qst_vocab_dict,
            ans_vocab_dict=ans_vocab_dict,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            shuffle=True,
        ),
        "valid": load_data_loader(
            data_path=valid_data_path,
            qst_vocab_dict=qst_vocab_dict,
            ans_vocab_dict=ans_vocab_dict,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            shuffle=True,
        ),
    }
