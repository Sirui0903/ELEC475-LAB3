import torch
import torch.nn as nn
import torchvision.models as models

class Vanilla(nn.Module):
    def __init__(self, num_classes=100):
        super(Vanilla, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])
        self.frontend = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = self.frontend(x)
        return x

