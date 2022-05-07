from torch import nn
import torchvision.models as models
import torch


class ImageChannel(nn.Module):
    def __init__(self, embed_size: int):
        """
        Args:
            embed_size(int): 이미지 체널의 out_features
        """
        super().__init__()
        model = models.vgg19(pretrained=True)
        in_features = models.classifier[-1].in_feature
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        self.model = model
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image(torch.Tensor): batch 크기만큼 들어있는 이미지 텐서 (shape=(batch_size, 3, 224, 224))
        Return:
            torch.Tensor
        """
        with torch.no_grad():
            image_features = self.model(image)
        image_features = self.fc(image_features)  # (batch_size, embed_size)

        l2_norm = image_features.norm(p=2, dim=1, keepdim=True)  # (batch_size)
        normalized = image_features.div(l2_norm)  # (batch_size, embed_size)

        return normalized


class TextChannel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class LSTM_VQA(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
