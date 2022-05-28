from typing import TypedDict, List, Dict, Optional
import torch.utils.data
from src.utils.vocab_dict import VocabDict
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
# batch_step_size = len(self.data_loader[phase].dataset) / batch_size


# TODO: pretrained된 이름 기반으로 코드 아래로 작성하고 인자로 잘 빼기
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', model_max_length=30)
is_image_preprocess = True

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
        question_token,
        answer_label=None,
        answer_multi_choice=None,
    ):
        self.image = image
        self.question = question
        self.question_token = question_token
        self.answer_label = answer_label
        self.answer_multi_choice = answer_multi_choice

    def to_dict(self):
        return {
            "image": self.image,
            "question": self.question,
            "question_token": self.question_token,
            "answer_label": self.answer_label,
            "answer_multi_choice": self.answer_multi_choice,
        }



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_tensor_load:bool,
        dataset: List[VQA_Raw_Data],
        vg_dataset,
        max_qst_length: int,
        max_num_ans: int,
        question_dict: VocabDict,
        answer_dict: Optional[VocabDict] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        """Visual Question Answering(vqa) + Visual Ground(vg) Dataset"""
        self.dataset = dataset
        self.vg_dataset = vg_dataset
        self.transform = transform
        self.image_tensor_load=image_tensor_load
        if len(dataset) > 0:
            self.load_ans = dataset[0].valid_answers is not None
            self.question_dict = question_dict
            self.answer_dict = answer_dict
            self.max_qst_length = max_qst_length
            self.max_num_ans = max_num_ans

    def __len__(self):
        """dataset에 들어있는 데이터 개수"""
        return len(self.dataset) + len(self.vg_dataset)

    def __getitem__(self, index: int) -> VQA_Input_Data:
        if index < len(self.dataset):
            vqa_data = self.dataset[index]
            image = 1
            if self.image_tensor_load == False:
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
            question_token = tokenizer(vqa_data.question_str, padding='max_length', truncation=True, return_tensors='pt', max_length = 30)
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
            result = VQA_Input_Data(
                image=image,
                question=quest_idx_list,
                question_token=question_token,
                answer_label=answer_label,
                answer_multi_choice=answer_multi_choice,
            ).to_dict()
            result["data_type"]="vqa"
            result["name"]=vqa_data.image_path
            return result
        else:
            vg_data = self.vg_dataset[index - len(self.dataset)]
            # preprocess image
            image_id = vg_data["image_id"]
            image = 1
            if not self.image_tensor_load:
                ori_image_rgb = Image.open(os.path.join('.','datasets','Images','train2014', f'COCO_train2014_{image_id:012}.jpg')).convert("RGB")
                image_rgb = Image.open(os.path.join('.','datasets','Resized_Images','train2014', f'COCO_train2014_{image_id:012}.jpg')).convert("RGB")
                image = self.transform(image_rgb) if self.transform else image_rgb
            sentence = vg_data["sentence"]
            question_token = tokenizer(sentence, padding='max_length', truncation=True, return_tensors='pt', max_length = 30)
            bbox = vg_data["bbox"]
            b0 = bbox[0] / ori_image_rgb.size[0] * 224
            b1 = bbox[1] / ori_image_rgb.size[1] * 224
            b2 = bbox[2] / ori_image_rgb.size[0] * 224
            b3 = bbox[3] / ori_image_rgb.size[1] * 224
            if b0 > 224 or b1 > 224:
                print(bbox)
                print(ori_image_rgb.size)
                print(image_id)
            bbox = torch.Tensor([b0,b1,b2,b3])

            result = dict(
                data_type="vg",
                image=image,
                sentence=sentence,
                question_token=question_token,
                bbox=bbox,
            )
            return result

    def get_vocab_size(self, vocab_type: str):
        if vocab_type == "question":
            return self.question_dict.vocab_size
        if vocab_type == "answer":
            return self.answer_dict.vocab_size
        raise Exception("invalid params (vocab_type)")

def load_vqa_data_loader(
    image_tensor_load:bool,
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
    vqa_data = [VQA_Raw_Data(x) for x in np.load(data_path, allow_pickle=True)]

    vqa_data = np.array(
        vqa_data[:len(vqa_data)]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize
    ])
    vqa_dataset = Dataset(
        image_tensor_load=image_tensor_load,
        dataset=vqa_data,
        vg_dataset=[],
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

def load_vg_data_loader(
    image_tensor_load:bool,
    vg_data_path:str,
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
    # vqa_data = [VQA_Raw_Data(x) for x in np.load(data_path, allow_pickle=True)]
    vg_data = np.load(vg_data_path, allow_pickle=True)

    # vqa_data = np.array(
    #     vqa_data[:len(vqa_data)//100]
    # )
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])
    vqa_dataset = Dataset(
        image_tensor_load=image_tensor_load,
        dataset=[],
        vg_dataset=vg_data,
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


# TODO: pretrained 이름을 받기


def load_VQA_DataLoader(
    image_tensor_load: bool,
    train_data_path: str,  #
    valid_data_path: str,  #
    train_vg_data_path: str,
    valid_vg_data_path: str,
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
        train_vg_data_path: train시      preprocess된 visual ground 데이터(train.npy)가 있는 경로
        valid_vg_data_path: validation시 preprocess된 visual ground 데이터(val.npy)가 있는 경로
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
        "train_vg": load_vg_data_loader(
            image_tensor_load=image_tensor_load,
            vg_data_path=train_vg_data_path,
            qst_vocab_dict=qst_vocab_dict,
            ans_vocab_dict=ans_vocab_dict,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            shuffle=True,
        ),
        "train_vqa": load_vqa_data_loader(
            image_tensor_load=image_tensor_load,
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
        "valid_vg": load_vg_data_loader(
            image_tensor_load=image_tensor_load,
            vg_data_path=valid_vg_data_path,
            qst_vocab_dict=qst_vocab_dict,
            ans_vocab_dict=ans_vocab_dict,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            shuffle=True,
        ),
        "valid_vqa": load_vqa_data_loader(
            image_tensor_load=image_tensor_load,
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
