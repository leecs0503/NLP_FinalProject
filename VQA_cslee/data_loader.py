import torch.utils.data as data
from typing import List
from PIL import Image
from util.text_helper import VocabDict


class VQA_Input_Data():
    def __init__(self, vqa_data: dict):
        self.image_name = vqa_data.image_name,
        self.image_path = vqa_data.image_path,
        self.question_id = vqa_data.question_id,
        self.question_str = vqa_data.question_str,
        self.question_tokens = vqa_data.question_tokens
        self.all_answers = vqa_data.all_answers if "all_answers" in vqa_data else None
        self.valid_answers = vqa_data.valid_answers if "valid_answers" in vqa_data else None


class VQA_Data():
    def __init__(
            self,
            image,
            question,
            answer=None
    ):
        self.image = image
        self.question = question
        self.answer = answer


class VQA_Dataset(data.Dataset):
    def __init__( 
            self,
            dataset: List[VQA_Input_Data],
            transform=None,
            question_dict: VocabDict = None,
            answer_dict: VocabDict = None
    ):
        """ VQA의 Dataset
        """
        self.dataset = dataset
        self.load_ans = dataset[0].valid_answers is not None
        self.question_dict = question_dict
        self.transform = transform



    def __len__(self):
        """ dataset에 들어있는 데이터 개수
        """
        return len(self.dataset)


    def __getitem__(self, index: int):
        vqa_data = self.dataset[index]

        image_rgb = Image.open(vqa_data.image_path).convert("RGB")
        print(type(image_rgb))

        return VQA_Data(
            image=image_rgb,
            question=1,
            answer=1
        )


def get_vqa_data_loader(
        train_data_path: str,   # 
        valid_data_path: str,   # 
        qst_vocab_path: str,    #
        ans_vocab_path: str,    #
        max_qst_length: int,    # 
        max_num_ans: int,       # 
        batch_size: int,        # 
        num_workers: int        #
):
    pass
