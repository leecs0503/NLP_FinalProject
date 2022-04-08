import torch.utils.data as data
from typing import List


class VQA_Data():
    def __init__(self, vqa_data: dict):
        self.image_name = vqa_data.image_name,
        self.image_path = vqa_data.image_path,
        self.question_id = vqa_data.question_id,
        self.question_str = vqa_data.question_str,
        self.question_tokens = vqa_data.question_tokens
        self.all_answers = vqa_data.all_answers
        self.valid_answers = vqa_data.valid_answers


class VQA_Dataset(data.Dataset):
    def __init__(self, dataset: List[VQA_Data]):
        self.dataset = dataset
        pass


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        vqa_data = self.dataset[index]
        image_name = vqa_data.image_name
        image_path = vqa_data.image_name
        question_id = vqa_data.image_name
        question_str = vqa_data.image_name
        question_tokens = vqa_data.image_name
        image_name = vqa_data.image_name


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):
    pass
