import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # (B, 2048)
        return x