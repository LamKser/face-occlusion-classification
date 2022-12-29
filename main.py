from argparse import ArgumentParser
from run import Run

parser = ArgumentParser(description='Face occlusion')

parser.add_argument('--train-path', default=None, type=str,
                    help='Train set (default: None)')
parser.add_argument('--val-path', default=None, type=str,
                    help='Validation set default: None)')
parser.add_argument('--test-path', default=None, type=str,
                    help='Test set (default: None)')
parser.add_argument('--save-dir', default=None, type=str,
                    help='Save weight directory (default: None)')
parser.add_argument('--weight-name', default='densenet169.pt', type=str,
                    help='Weight file (default: "resnet18.pt")')
parser.add_argument('--csv', default='densenet169.csv', type=str,
                    help='Save csv (default: "densenet169.csv")')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='Pretrained (ImageNet) (default: True)')
parser.add_argument('--mode', default='train', type=str,
                    help='Choose mode for running model (default: train)')
parser.add_argument('--logger-path', default='logger', type=str,
                    help='Logger path (default: runs')

args = parser.parse_args()

if __name__ == "__main__":
    model = Run(pretrained=args.pretrained)
    if args.mode == 'train':
        model.train(args.train_path, args.val_path, args.save_dir, args.weight_name, args.logger_path)

    elif args.mode == 'test':
        model.test(args.test_path, args.save_dir, args.weight_name, args.csv)

