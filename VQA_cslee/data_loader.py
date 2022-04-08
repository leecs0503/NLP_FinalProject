import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper


class VQA_Dataset(data.Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):
    pass