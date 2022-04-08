import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config):
    pass


def train():
    pass


def define_argparse():
    p = argparse.ArgumentParse()

    p.add_arguments('--image-model-name', type=str, help='image model name (one of [\'vgg19\'])')

    config = p.parse_args()
    return config


if __name__ == '__main__':
    config = define_argparse()
    main(config)
