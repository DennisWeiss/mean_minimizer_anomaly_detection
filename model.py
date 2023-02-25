import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.projection_size = 256

        # self.backbone = torchvision.models.resnet18(pretrained=False)

        # self.feature_size = self.backbone.fc.in_features

        # self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        # self.backbone.maxpool = nn.Identity()
        # self.backbone.fc = nn.Identity()

        self.feature_size = 512

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.feature_size),
            nn.ReLU()
        )

        self.projector = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            nn.Linear(1024, self.projection_size),
        )

    def forward(self, x):
        # return self.backbone(x)
        return self.projector(self.backbone(x))