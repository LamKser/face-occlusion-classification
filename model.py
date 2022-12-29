from torchvision import models
from torch import nn
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Model(nn.Module):

    def __init__(self, name: str, num_class: int, pretrained: bool):
        super(Model, self).__init__()
        
        # ResNet 18
        if name == 'resnet18':
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18()

        # ResNet 50
        if name == 'resnet50':
            if pretrained:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet50()

        # DenseNet 169
        elif name == 'densenet169':
            if pretrained:
                self.model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
            else:
                self.model = models.densenet169()

        # Change the number of class
        if 'resnet' in name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)
        elif 'densenet' in name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_class)

        print(f'Model: {name}')

    def forward(self, x):
        return self.model(x)
