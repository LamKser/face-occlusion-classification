from torchvision import transforms
import torch
from PIL import Image
import onnxruntime

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = [224, 224]
CLASSES = {0: "non-occluded",
           1: "occluded"}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_image(image):
    transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
    ])
    img = Image.open(image)
    img = transform(img)
    return img.unsqueeze(0)


def run_onnx_model(onnx_model, pil_image):
    # x = torch.randn(1, 3, 224, 224, requires_grad=True)
    ort_session = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(pil_image)}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    
    output = torch.softmax(torch.from_numpy(ort_outs), 1)
    prob, pred = torch.max(output, 1)
    return prob.item(), pred.item()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True,
                        help="onnx model")
    parser.add_argument("--img", type=str, required=True, 
                        help="Your image")
    args = parser.parse_args()

    img = load_image(args.img)
    model = run_onnx_model(args.onnx, img)
    po, pe = model
    print(f"Face is {CLASSES[pe]} with {po * 100 :.2f}")
