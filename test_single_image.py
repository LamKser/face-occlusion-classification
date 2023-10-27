from PIL import Image
import torch
from torchvision import transforms

from utils import load_weight
from model import Model


# CONSTANT
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = [224, 224]
CLASSES = {0: "non-occluded",
           1: "occluded"}


def test_image(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(opt.model, 2, False).to(device)
    model = load_weight(model, opt.weight)
    model.eval()

    # transform data
    transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
    ])

    # Image
    img = Image.open(opt.image)
    img = transform(img).to(device)
    output = model(img.unsqueeze(0))
    output = torch.softmax(output, 1)
    prob, pred = torch.max(output, 1)
    
    print("Image {} is {} - {:.2f} %".format(
        opt.image, CLASSES[pred.item()], prob.item() * 100
    ))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--weight", type=str, help="Weight path (.pth)")
    parser.add_argument("--image", type=str, help="Image path")
    args = parser.parse_args()

    test_image(args)
