import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class VQADataset(data.Dataset):
    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):
    pass