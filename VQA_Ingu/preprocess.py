import os
import argparse
import numpy as np
import torch
from util import text_helper
from util import text_preprocess
from util import image_preprocess
from util import input_maker


def main(args):
    pass


if __name__ == 'main':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_input_dir',
        type=str,
        default='/datasets/Images',
        help='directory for input images (unresized images)'
    )

    parser.add_argument(
        '--image_output_dir',
        type=str,
        default='/datasets/Resized_Images',
        help='directory for output images (resized images)'
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='size of images after resizing'
    )

    parser.add_argument(
        '--text_input_dir',
        type=str,
        default='/datasets',
        help='directory for input questions and answers'
    )

    parser.add_argument(
        '--n_answers',
        type=int,
        default=1000,
        help='the number of answers to be kept in vocab'
    )

    parser.add_argument(
        '--vqa_output_dir',
        type=str,
        default='/datasets',
        help='directory for outputs'
    )

    args = parser.parse_args()

    main(args)
