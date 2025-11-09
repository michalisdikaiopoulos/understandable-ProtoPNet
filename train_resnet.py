import argparse
from ppnet.tetresnet import run_resnet_training

# Script arguments
parser = argparse.ArgumentParser(
    description='Train a ResNet baseline model',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42)
)

# Required arguments
parser.add_argument('--dataset', type=str, required=True,
                    help='path of the dataset to use for training and evaluation')
parser.add_argument('--exp_name', type=str, required=True,
                    help='id of the current experiment')

# Model architecture
parser.add_argument('--architecture', type=str, default='resnet34',
                    help='ResNet architecture to use (default: %(default)s)',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])

# Training parameters
parser.add_argument('--epochs', type=int, default=10000,
                    help='number of training epochs (default: %(default)s)')
parser.add_argument('--img_size', type=int, default=224,
                    help='resize dimension for training images (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size (default: %(default)s)')
parser.add_argument('--step_size', type=int, default=150,
                    help='step size of the learning rate scheduler (default: %(default)s)')
parser.add_argument('--test_interval', type=int, default=30,
                    help='epoch interval in which to run the model on the test split (default: %(default)s)')

# System parameters
parser.add_argument('--gpus', type=str, default='0',
                    help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers to use for data loading (default: %(default)s)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed to use (default: %(default)s)')

if __name__ == '__main__':
    # Start training
    args = parser.parse_args()
    run_resnet_training(args)
