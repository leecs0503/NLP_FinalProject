# import torch
# import torch.nn as nn
# import torchvision.models as models
from typing import Protocol, Dict, List


class Model(Protocol):
    """Ducktyping을 위한 Model의 Protocol Class"""

    def __init__(self, name: str) -> str:
        pass

    def get_params(self) -> List:
        """
        Return: list
        """
        pass

    def state_dict(self) -> Dict:
        """현재 모델의 state_dict를 반환하는 메소드

        Return:
            state_dict
        """
        pass

    def load_state_dict(self, state_dict):
        """state_dict로 부터 모델의 parameter을 불러오는 메소드

        Args:
            dict: 현재 모델로 부터 state_dict메소드를
        """
        pass

    def get_name(self) -> str:
        """모델의 이름을 가져오는 메소드

        Returns:
            str: 현재 Model의 이름
        """
        pass

    def forward(self, **kwargs):
        """데이터를 받아서 텐서를 반환
        Args:
            kwargs: 현재 모델에 맞는 데이터들

        Returns:
            torch.Tensor: 모델이 반환한 텐서
        """
        pass
