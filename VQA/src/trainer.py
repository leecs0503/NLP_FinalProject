from datetime import datetime
import logging
from src.model.model import Model
from src.dataloader import VQA_DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim
from typing import Literal
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from typing import List
from warmup_scheduler import GradualWarmupScheduler


def acc_multi_choice(
    pred_exp: torch.Tensor, answer: List[torch.Tensor]
) -> torch.Tensor:
    """multi-choice방식에서 accuracy를 계산하는 메소드

    Args:
        pred_exp: 모델이 예측한 결과 (batch_size)
        answer: 실제 결과 (num_answer, batch_size)
    Returns:
        torch.Tensor: 계산된 acc점수의 합
    """
    return torch.stack([(ans == pred_exp.cpu()) for ans in answer]).any(dim=0).sum()


def acc_open_ended(pred_exp: torch.Tensor, answer: List[torch.Tensor]) -> torch.Tensor:
    """open-ended방식에서 accuracy를 계산하는 메소드

    Args:
        pred_exp: 모델이 예측한 결과 (batch_size)
        answer: 실제 결과 (num_answer, batch_size)
    Returns:
        torch.Tensor: 계산된 acc점수의 합
    """
    return (
        torch.stack([ans == pred_exp.cpu() for ans in answer])
        .sum(dim=0)
        .div(3)
        .minimum(torch.ones(answer[1].shape[0]))
        .sum()
    )


class VQA_Trainer:
    """ """

    def __init__(
        self,
        device: torch.device,
        model: Model,
        data_loader: VQA_DataLoader,
        model_path: str,
        log_path: str,
        tensorboard_path: str = "runs/",
    ):
        # todo: device를 context로
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.model_path = model_path
        self.writer = SummaryWriter(tensorboard_path)
        self.logger = logging.getLogger("VQA_Trainer logger")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(
            os.path.join(log_path, str(datetime.now()) + ".log")
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.prev_acc = 0 # hyperparameter tuning을 위한 변수

    def load_model(
        self,
        start_epochs: int,
    ):
        load_path = os.path.join(
            self.model_path, f"{self.model.get_name()}-epoch-{start_epochs:02}.ckpt"
        )
        self.model.load_state_dict(torch.load(load_path)["state_dict"])

    def save_model(
        self,
        epoch: int,
    ):
        save_path = os.path.join(
            self.model_path, f"{self.model.get_name()}-epoch-{epoch:02}.ckpt"
        )
        torch.save(
            {"epoch": epoch + 1, "state_dict": self.model.state_dict()}, save_path
        )

    def log_batch(
        self,
        loss: float,
        corr_exp: float,
        batch_size: float,
        phase: Literal["train", "valid"],
        num_epochs,
        epoch: int,
        batch_idx: int,
    ):
        self.writer.add_scalar(f"Step{epoch:02}/Loss/{phase.upper()}-{epoch:02}", loss, batch_idx)
        self.writer.add_scalar(f"Step{epoch:02}/ACC/{phase.upper()}-{epoch:02}", corr_exp/batch_size, batch_idx)
        self.writer.flush()
        msg = "| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}".format(
            phase.upper(),
            epoch + 1,
            num_epochs,
            batch_idx,
            int(
                len(self.data_loader[phase].dataset)
                / self.data_loader[phase].batch_size
            ),
            loss.item(),
        )
        print(msg)
        self.logger.info(msg)

    def log_step(
        self,
        epoch_loss: float,
        epoch_acc_exp: float,
        phase: Literal["train", "valid"],
        epoch: int,
        num_epochs: int,
    ):
        self.writer.add_scalar(f"Epoch/Loss/{phase.upper()}", epoch_loss, epoch)
        self.writer.add_scalar(f"Epoch/ACC/{phase.upper()}", epoch_acc_exp, epoch)
        self.writer.flush()
        msg = f"| {phase.upper()} SET | Epoch [{epoch + 1:02}/{num_epochs:02}], Loss: {epoch_loss:.4}, Acc(Exp): {epoch_acc_exp:.4}"
        print(msg)
        self.logger.info(msg)
        pass

    def step(
        self,
        epoch: int,
        num_epochs: int,
        criterion: nn.CrossEntropyLoss,
        optimizer, # optim.Adam
        scheduler: GradualWarmupScheduler,
    ):
        for phase in self.data_loader:  # equal to: for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corr_exp = 0
            batch_step_size = 0
            running_count = 0

            optimizer.zero_grad()

            if phase == "train":
                optimizer.step()
                scheduler.step(epoch)
                self.model.train()
            else:
                self.model.eval()

            for batch_idx, batch_sample in enumerate(self.data_loader[phase]):
                # todo: 아래 로직을 함수로 빼기

                image = batch_sample["image"].to(self.device)
                question = batch_sample["question"].to(self.device)
                question_token = batch_sample["question_token"].to(self.device)
                label = batch_sample["answer_label"].to(self.device)
                multi_choice = batch_sample["answer_multi_choice"]  # not tensor, list.

                batch_step_size += 1

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = self.model(
                        image, question_token
                    )  # [batch_size, ans_vocab_size=1000]
                    _, pred_exp = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # pred_exp[pred_exp == ans_unk_idx] = -9999
                running_loss += loss.item()
                corr_exp = acc_open_ended(pred_exp, multi_choice)
                batch_size = question.shape[0]
                running_corr_exp += corr_exp
                running_count += question.shape[0]
                self.log_batch(
                    loss=loss,
                    corr_exp=corr_exp,
                    batch_size=batch_size,
                    phase=phase,
                    num_epochs=num_epochs,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp = running_corr_exp.double() / len(
                self.data_loader[phase].dataset
            )
            self.log_step(
                epoch_loss=epoch_loss,
                epoch_acc_exp=epoch_acc_exp,
                phase=phase,
                epoch=epoch,
                num_epochs=num_epochs,
            )
            if self.prev_acc > epoch_acc_exp:
                early_stop = True
            else:
                early_stop = False
            self.prev_acc = epoch_acc_exp
            return early_stop

    def run(
        self,
        optimizer,
        step_size: int,
        gamma: float,
        save_step: int = 1,
        start_epochs: int = 0,
        num_epochs: int = 30,
    ):
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_warmup = GradualWarmupScheduler(optimizer=optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler)

        if start_epochs != 0:
            self.load_model(start_epochs)

        for epoch in range(start_epochs, num_epochs):
            early_stop = self.step(
                epoch=epoch,
                num_epochs=num_epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler_warmup,
            )
            if (epoch + 1) % save_step == 0:
                self.save_model(epoch)
            if early_stop == True:
                break
