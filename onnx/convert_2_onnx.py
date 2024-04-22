import sys, os
sys.path.append(os.getcwd())

import torch
import onnx

from utils import load_weight
from model import Model

def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch.
       Ref: yolov8
    """
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1  # opset

def check_dir_exist(path):
    path_split = os.path.split(path)
    if not path_split[0]:
        return path_split[-1]
    else:
        if not os.path.exists(path_split[0]):
            os.makedirs(path_split[0])
        return path


class Convert2ONNX:

    def __init__(self, model_name, weight) -> None:
        print(f"Convert model {model_name} to ONNX")

        self.model_name = model_name
        self.model = Model(model_name, 2, is_train=False)
        self.model = load_weight(self.model, weight)
    
    def __call__(self, save, opset_version=None):
        path = check_dir_exist(save)
        # Export model
        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            verbose=True,
            export_params=True,
            opset_version=opset_version if opset_version else get_latest_opset(),
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ["input"],   # the model"s input names 
            output_names = ["output"], # the model"s output names 
            dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes 
                          "output" : {0 : "batch_size"}}
        )
        print(f"Model {self.model_name} is converted to ONNX")

        # Check converted model
        onnx_model = onnx.load(save)
        onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                      help="Model name (resnet | densenet | vgg | )")
    parser.add_argument("--weight", type=str, required=True,
                      help='Weight of model ({"state_dict":"", "epoch": ""})')
    parser.add_argument("--save", type=str, default="onnx/model.onnx",
                      help='Save path (default: "onnx/model.onnx")')
    parser.add_argument("--opset_version", type=int, default=None,
                      help='Opset version (default: None)')
    args = parser.parse_args()

    pt2onnx = Convert2ONNX(args.model, args.weight)
    pt2onnx(args.save, args.opset_version)


    