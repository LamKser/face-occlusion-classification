from argparse import ArgumentParser
import yaml

from src.train import TrainModel
from src.eval import EvalModel
from src.test import TestModel


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mode", type=str, help='Enter mode (train/eval/test)')
    parser.add_argument("--configs-path", type=str, help='Enter train/test config yaml')
    parser.add_argument("--image-path", type=str, help='Path of image for testing')
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    if args.mode == "train":
        TrainModel(config).train()
    elif args.mode == "eval":
        EvalModel(config).val()
    elif args.mode == "test":
        TestModel(config).run(args.image_path)