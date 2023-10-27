from torch import nn
from PIL import ImageFile

from utils import get_model


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Model(nn.Module):

    def __init__(self, name: str, num_class: int, pretrained: bool = False):
        super(Model, self).__init__()
        
        self.model = get_model(name, pretrained)

        # Change the number of class
        if 'resnet' in name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)
        elif 'densenet' in name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_class)
        elif "vgg" in name:
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, num_class)
        elif "convnext" in name:
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, num_class)
        print(f'Model: {name}')

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Model("convnext_large", 2, True)
    print(model)