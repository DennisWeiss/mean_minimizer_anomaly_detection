import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.projection_size = 256

        # self.backbone = torchvision.models.resnet18(pretrained=False)
        #
        # self.feature_size = self.backbone.fc.in_features
        #
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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return self.classifier(x)


class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()

        self.projection_size = 256

        self.backbone = torchvision.models.resnet18(pretrained=False)

        self.feature_size = self.backbone.fc.in_features

        self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 32, 5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        #     nn.Linear(128 * 4 * 4, 256),
        #     nn.ReLU()
        # )

        self.projector = nn.Sequential(
            nn.Linear(self.feature_size, 2048),
            nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            nn.Linear(2048, self.projection_size),
        )

    def forward(self, x):
        # return self.backbone(x)
        return self.projector(self.backbone(x))