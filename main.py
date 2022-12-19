from argparse import ArgumentParser
from model import RunModel
import torch
import os

parser = ArgumentParser(description='Run VGG19')

parser.add_argument('--validation', default=True, type=bool,
                    help='Validate model (default: "True")')
parser.add_argument('--name', default='renet18', type=str,
                    help='Model name (default: "resnet18")')

# Data path
parser.add_argument('--train-path', default=None, type=str,
                    help='Training data path (default: None)')
parser.add_argument('--val-path', default=None, type=str,
                    help='Validation data path (default: None)')
parser.add_argument('--test-path', default=None, type=str,
                    help='Test data path (default: None)')
parser.add_argument('--save-path', default=None, type=str,
                    help='Save weight path (default: None)')
parser.add_argument('--weight-file', default='resnet18.pt', type=str,
                    help='Weight file (default: "resnet18.pt")')
parser.add_argument('--csv-file', default='resnet18_predict.csv', type=str,
                    help='Save score to csv (default: "resnet18_predict.csv")')

# Configure model
parser.add_argument('--num-class', default=2, type=int,
                    help='Number of class (default: 2) ')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size (default: 16)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate (default: 1e-3)')
parser.add_argument('--weight-decay', default=0.9, type=float,
                    help='weight decay (default: 0.9)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')

# Set scheduler
parser.add_argument('--is-scheduler', default=False, type=bool,
                    help='Is scheduler (default: False)')
parser.add_argument('--step-size', default=25, type=int,
                    help='Step size (default: 25)')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma (default: 0.1)')

parser.add_argument('--pretrained', default=False, type=bool,
                    help='Pretrained (ImageNet) (default: False)')
parser.add_argument('--epochs', default=15, type=int,
                    help='Epochs (default: 20)')
parser.add_argument('--mode', default='train', type=str,
                    help='Choose mode for running model (default: train)')
parser.add_argument('--logger-path', default='runs', type=str,
                    help='Logger path (default: runs')
parser.add_argument('--continue-train', default=False, type=bool,
                    help='Continue train model (default: False)')
args = parser.parse_args()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = RunModel(device=device, name=args.name,
                   train_path=args.train_path, val_path=args.val_path, test_path=args.test_path, batch_size=args.batch_size,
                   lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                   is_scheduler=args.is_scheduler, step_size=args.step_size, gamma=args.gamma,
                   num_class=args.num_class, pretrained=args.pretrained)

    if args.mode == 'train':
        run.train(args.epochs, args.save_path, args.weight_file, args.logger_path, args.validation, args.continue_train)

    elif args.mode == 'test':
        run.test(args.csv_file, os.path.join(args.save_path, args.weight_file))

