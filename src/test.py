import yaml
from io import BytesIO

from PIL import Image
import torch
from torchvision import transforms

from src.model import Model
from src.utils import load_weight


# CONSTANT
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]
# SIZE = [224, 224]
CLASSES = {0: "non-occluded",
           1: "occluded"}

class TestModel:

    def __init__(self, config):

        # Load config
        # with open(config_path, 'r') as f:
        #     self.config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        self.save = self.config["save"]
        self.data = self.config["data"]

        # Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)

        if (self.config['model']["classifier"] == "sigmoid") \
            and (self.config["train"]['num_class'] != 1):
            self.config["train"]['num_class'] = 1

        model = Model(self.config['model']["name"], 
                           self.config["data"]['num_class'], 
                           False).to(self.device)
        self.model = load_weight(model, self.config["weight"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(self.data["size"]),
            transforms.ToTensor(),
            transforms.Normalize(self.data['mean'], self.data['std'])
        ])

    def run(self, image_path):
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = Image.open(BytesIO(image_path.getvalue()))
        img = self.transform(img).to(self.device)
        output = self.model(img.unsqueeze(0))
        # output = torch.softmax(output, 1)
        # prob, pred = torch.max(output, 1)

        if self.config['model']["classifier"] == "softmax":
            probs = torch.softmax(output, 1)
            prob, predict = torch.max(probs, 1)
        elif self.config['model']["classifier"] == "sigmoid":
            prob = torch.sigmoid(output)
            predict = probs.round()
        print(f"Image \033[1;33m{image_path}\033[0m is \033[1;33m{CLASSES[predict.item()]}\033[0m - \033[1;33m{prob.item() * 100:.2f} %\033[0m")
        return prob, predict
