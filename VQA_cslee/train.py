import torch
import argparse
import os
from data_loader import get_vqa_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    # directory setting
    input_dir: str,
    log_dir: str,
    model_dir: str,
    # data_loader setting
    max_qst_length: int,
    max_num_ans: int,
    batch_size: int,
    num_workers: int,
    # model setting
    image_model_name: str,
    # train setting
    num_epoch: int,
):
    """VQA model에 대해 train"""

    # 학습 환경 설정
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # get_loader
    _ = get_vqa_data_loader(
        train_data_path=os.path.join(input_dir, "train.npy"),
        valid_data_path=os.path.join(input_dir, "valid.npy"),
        qst_vocab_path=os.path.join(input_dir, "vocab_questions.txt"),
        ans_vocab_path=os.path.join(input_dir, "vocab_answers.txt"),
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 학습 Epoch 시작


def main(args):
    train(
        # directory setting
        input_dir=args.input_dir,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        # data_loader setting
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # model setting
        image_model_name=args.image_model_name,
        # train setting
        num_epoch=args.num_epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory setting
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./datasets",
        help="folder which contains test, traia, valid.npy, etc",
    )

    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="directory for logs."
    )

    parser.add_argument(
        "--model_dir", type=str, default="./models", help="directory for saved models."
    )

    # data_loader setting
    parser.add_argument("--max_qst_length", type=int, default=30)
    parser.add_argument("--max_num_ans", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # data_loader setting

    parser.add_argument(
        "--image_model_name", type=str, help="image model name (one of ['vgg19'])"
    )

    # train setting
    parser.add_argument("--num_epoch", type=int, default=30)

    args = parser.parse_args()
    main(args)
