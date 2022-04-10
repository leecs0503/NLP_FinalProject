import torch
import argparse
import os
from models import VQAModel
from data_loader import get_vqa_data_loader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

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
    embed_size: int,
    word_embed_size: int,
    num_layers: int,
    hidden_size: int,
    # hyper parameter setting
    learning_rate: float,
    step_size: int,
    gamma: float,
    # train setting
    num_epochs: int,
    save_step: int
):
    """VQA model에 대해 train"""

    # 학습 환경 설정
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # get_loader
    data_loader = get_vqa_data_loader(
        train_data_path=os.path.join(input_dir, "train.npy"),
        valid_data_path=os.path.join(input_dir, "valid.npy"),
        qst_vocab_path=os.path.join(input_dir, "vocab_questions.txt"),
        ans_vocab_path=os.path.join(input_dir, "vocab_answers.txt"),
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    qst_vocab_size = data_loader["train"].dataset.get_vocab_size("question")
    ans_vocab_size = data_loader["train"].dataset.get_vocab_size("answer")
    ans_unk_idx = data_loader["train"].dataset.answer_dict.unk2idx

    # model 준비
    model = VQAModel(
        image_model_name=image_model_name,
        embed_size=embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=word_embed_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
    ).to(device)

    # 학습 알고리즘 설정
    params = (
        list(model.img_encoder.fc.parameters())
        + list(model.qst_encoder.parameters())
        + list(model.fc1.parameters())
        + list(model.fc2.parameters())
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 학습 Epoch 시작

    for epoch in range(num_epochs):
        for phase in ["train", "valid"]:
            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / batch_size

            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                image = batch_sample["image"].to(device)
                question = batch_sample["question"].to(device)
                label = batch_sample["answer_label"].to(device)
                multi_choice = batch_sample["answer_multi_choice"]  # not tensor, list.

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    output = model(image, question)      # [batch_size, ans_vocab_size=1000]
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                running_corr_exp1 += (
                    torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice])
                    .any(dim=0)
                    .sum()
                )
                running_corr_exp2 += (
                    torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice])
                    .any(dim=0)
                    .sum()
                )

                # Print the average loss in a mini-batch.
                if batch_idx % 1 == 0:
                    print(
                        "| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}".format(
                            phase.upper(),
                            epoch + 1,
                            num_epochs,
                            batch_idx,
                            int(batch_step_size),
                            loss.item(),
                        )
                    )

            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(
                data_loader[phase].dataset
            )  # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(
                data_loader[phase].dataset
            )  # multiple choice

            print(
                "| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n".format(
                    phase.upper(),
                    epoch + 1,
                    num_epochs,
                    epoch_loss,
                    epoch_acc_exp1,
                    epoch_acc_exp2,
                )
            )

            # Log the loss and accuracy in an epoch.
            with open(
                os.path.join(log_dir, "{}-log-epoch-{:02}.txt").format(
                    phase, epoch + 1
                ),
                "w",
            ) as f:
                f.write(
                    str(epoch + 1)
                    + "\t"
                    + str(epoch_loss)
                    + "\t"
                    + str(epoch_acc_exp1.item())
                    + "\t"
                    + str(epoch_acc_exp2.item())
                )

        # Save the model check points.
        if (epoch + 1) % save_step == 0:
            torch.save(
                {"epoch": epoch + 1, "state_dict": model.state_dict()},
                os.path.join(model_dir, "model-epoch-{:02d}.ckpt".format(epoch + 1)),
            )


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
        embed_size=args.embed_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        # hyper parameter setting
        learning_rate=args.learning_rate,
        step_size=args.step_size,
        gamma=args.gamma,
        # train setting
        num_epochs=args.num_epochs,
        save_step=args.save_step
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    # model setting
    parser.add_argument(
        "--image_model_name", type=str, default="vgg19", help="image model name (one of ['vgg19'])"
    )

    parser.add_argument(
        "--embed_size",
        type=int,
        default=1024,
        help="embedding size of feature vector \
                              for both image and question.",
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
        "--hidden_size", type=int, default=512, help="hidden_size in the LSTM."
    )

    # hyper parameter setting
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate for training."
    )

    parser.add_argument(
        "--step_size", type=int, default=10, help="period of learning rate decay."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate decay.",
    )

    # train setting
    parser.add_argument("--num_epochs", type=int, default=30)

    parser.add_argument("--save_step", type=int, default=1, help="save step of model.")

    args = parser.parse_args()
    main(args)
