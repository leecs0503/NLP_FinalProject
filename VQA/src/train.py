import argparse
import torch
import os
import sys
from model.VGG19_Tansformer import Transformer_VQA

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.utils.vocab_dict import VocabDict
from src.dataloader import VQA_DataLoader, load_VQA_DataLoader
from src.trainer import VQA_Trainer
from src.model.model import Model
from src.model.VGG19_LSTM import LSTM_VQA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    log_dir: str,
    model_dir: str,
    tensorboard_dir: str,
    model: Model,
    data_loader: VQA_DataLoader,
    gamma: float,
    learning_rate: float,
    step_size: int,
    num_epochs: int,
    save_step: int,
    load_step: int,
):
    """모델을 학습시키는 메소드

    Args:
        log_dir: log파일을 저장할 경로
        model_dir: 학습된 모델을 저장하거나 불러올 경로
        tensorboard_dir : 텐서보드 경로
        model : 학습할 모델
        data_loader : 사용할 data loader
        gamma : learning rate decay의 가중치
        learning_rate : 학습시의 learning rate,
        step_size : laerning rate decay의 주기
        num_epochs : 학습할 epoch 수
        save_step : 모델 저장 주기
        load_step : 불러올 모델의 epoch값
    """

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # os.makedirs(tensorboard_dir, exist_ok=True)

    trainer = VQA_Trainer(
        device=device,
        model=model,
        data_loader=data_loader,
        model_path=model_dir,
        log_path=log_dir,
        tensorboard_path=tensorboard_dir,
    )

    trainer.run(
        learning_rate=learning_rate,
        step_size=step_size,
        gamma=gamma,
        save_step=save_step,
        start_epochs=load_step,
        num_epochs=num_epochs,
    )


def get_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # directories

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./datasets",
        help="directory for input",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="directory for logs",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="directory for save/load model",
    )

    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="directory for tensorboard",
    )

    # model setting

    parser.add_argument(
        "--model_name",
        type=str,
        default="VGG19_Transformer",
        help="The name of the model",
    )

    parser.add_argument(
        "--embed_size",
        type=int,
        default=1024,
        help="embedding size of feature vector \
        for both imageand question.",
    )

    parser.add_argument(
        "--word_embed_size",
        type=int,
        default=300,
        help="embedding size of word \
        used for the input in the LSTM.",
    )

    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of layers of the RNN(LSTM)."
    )

    parser.add_argument(
        "--hidden_layer_size",
        type=int,
        default=512,
        help="hidden layer size in the LSTM.",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="dropout rate of dropout layer.",
    )

    # transformer setting

    parser.add_argument(
        "--num_head", type=int, default=8, help="transformer's number of head"
    )

    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="transformer's feedforward layer's dimension",
    )

    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=6,
        help="transformer's number of encoder layer",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate decay.",
    )

    parser.add_argument("--batch_size", type=int, default=256, help="batch_size.")

    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of processes working on cpu."
    )

    parser.add_argument(
        "--max_question_length",
        type=int,
        default=30,
        help="maximum length of question. \
        the length in the VQA dataset = 26.",
    )

    parser.add_argument(
        "--max_num_ans", type=int, default=10, help="maximum number of answers."
    )

    # trainer setting

    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate for training."
    )

    parser.add_argument(
        "--step_size", type=int, default=10, help="period of learning rate decay."
    )

    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs.")

    parser.add_argument("--save_step", type=int, default=1, help="save step of model.")

    parser.add_argument("--load_step", type=int, default=0, help="load saved model.")

    return parser.parse_args()


def main():
    args = get_argument()

    data_loader = load_VQA_DataLoader(
        train_data_path=os.path.join(args.input_dir, "train.npy"),
        valid_data_path=os.path.join(args.input_dir, "valid.npy"),
        qst_vocab_dict=VocabDict(os.path.join(args.input_dir, "vocab_questions.txt")),
        ans_vocab_dict=VocabDict(os.path.join(args.input_dir, "vocab_answers.txt")),
        max_qst_length=args.max_question_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    qst_vocab_size = data_loader["train"].dataset.question_dict.vocab_size
    ans_vocab_size = data_loader["train"].dataset.answer_dict.vocab_size

    if args.model_name == "VGG19_LSTM":
        model = LSTM_VQA(
            embed_size=args.embed_size,
            qst_vocab_size=qst_vocab_size,
            word_embed_size=args.word_embed_size,
            num_layers=args.num_layers,
            hidden_size=args.hidden_layer_size,
            ans_vocab_size=ans_vocab_size,
            dropout_rate=args.dropout_rate,
        ).to(device)
    elif args.model_name == "VGG19_Transformer":
        model = Transformer_VQA(
            ans_vocab_size=ans_vocab_size,
            dropout_rate=args.dropout_rate,
            qst_vocab_size=qst_vocab_size,
            pad_token=0,
            embed_size=args.embed_size,
            hidden_size=args.hidden_layer_size,
            num_head=args.num_head,
            dim_feedforward=args.dim_feedforward,
            num_encode_layers=args.num_encoder_layers,
            max_qst_length=args.max_question_length,
        ).to(device)
    else:
        assert False
    train_model(
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        tensorboard_dir=args.tensorboard_dir,
        model=model,
        data_loader=data_loader,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        step_size=args.step_size,
        num_epochs=args.num_epochs,
        save_step=args.save_step,
        load_step=args.load_step,
    )


if __name__ == "__main__":
    main()
