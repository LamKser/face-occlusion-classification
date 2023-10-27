from argparse import ArgumentParser
from train_val_test import Run

parser = ArgumentParser(description='Face occlusion')


parser.add_argument('--image-path', type=str, help='Single image path')
parser.add_argument('--mode', default='train', type=str,
                    help='Choose mode for running model (default: train)')

args = parser.parse_args()

if __name__ == "__main__":
    model = Run()
    if args.mode == 'train':
        model.train()

    elif args.mode == 'test':
        model.test()

    # elif args.mode == 'image':
    #     model.test_image(args.image_path, args.weight_dir, args.weight_name)