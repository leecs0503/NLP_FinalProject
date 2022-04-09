import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from util import text_helper


class VQADataset(data.Dataset):
    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(os.path.join(input_dir, input_vqa), allow_pickle=True)
        self.question_vocabulary = text_helper.VocabDict(os.path.join(input_dir, 'vocab_questions.txt'))
        self.answer_vocabulary = text_helper.VocabDict(os.path.join(input_dir, 'vocab_answers.txt'))
        self.max_question_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.vqa[idx]['image_path']).convert('RGB')
        qst2idc = np.array([self.question_vocabulary.word2idx('<pad>')] * self.max_question_length)
        qst2idc[:len(self.vqa[idx]['question_tokens'])] = [self.question_vocabulary.word2idx(w) for w in self.vqa[idx]['question_tokens']]

        sample = {'image' : image, 'question' : qst2idc}

        if self.load_ans:
            ans2idc = [self.answer_vocabulary.word2idx(w) for w in self.vqa[idx]['valid_answers']]
            sample['answer_label'] = np.random.choice(ans2idc)
            mul2idc = [-1] * self.max_num_ans
            mul2idc[:len(ans2idc)] = ans2idc
            sample['answer_multi_choice'] = mul2idc

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def __len__(self):
        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    vqa_dataset = {
        'train' : VQADataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform
        ),
        'valid' : VQADataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform
        )
    }

    data_loader = {
        phase : torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        for phase in ['train', 'valid']
    }

    return data_loader
