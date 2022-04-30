from typing import TypedDict
import torch.utils.data

# batch_step_size = len(self.data_loader[phase].dataset) / batch_size


class VQA_DataLoader(TypedDict):
    train: torch.utils.data.DataLoader
    valid: torch.utils.data.DataLoader
