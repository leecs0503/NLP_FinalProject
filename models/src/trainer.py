from model import Model
from dataloader import VQA_DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class VQA_Trainer():
    """
    """
    def __init__(
        self,
        device: torch.device,
        model: Model,
        data_loader: VQA_DataLoader,
        model_path: str,
    ):
        # todo: device를 context로
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.model_path = model_path

    def load_model(
        self,
        start_epochs: int,
    ):
        load_path = os.path.join(self.model_path, f"{self.model.get_name()}-epoch-{start_epochs:02}.ckpt")
        self.model.load_state_dict(torch.load(load_path)["state_dict"])

    def save_model(
        self,
        epoch: int,
    ):
        save_path = os.path.join(self.model_path, f"{self.model.get_name()}-epoch-{epoch:02}.ckpt")
        torch.save({"epoch": epoch + 1, "state_dict": self.model.state_dict()}, save_path)

    def step(
        self,
        epoch: int,
        num_epochs: int,
        batch_size: int,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        scheduler: lr_scheduler.StepLR,
    ):
        for phase in self.data_loader:  # equal to: for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0

            if phase == "train":
                scheduler.step()
                self.model.train()
            else:
                self.model.eval()

            for batch_idx, batch_sample in enumerate(self.data_loader[phase]):
                # todo: 아래 로직을 함수로 빼기

                image = batch_sample["image"].to(self.device)
                question = batch_sample["question"].to(self.device)
                label = batch_sample["answer_label"].to(self.device)
                multi_choice = batch_sample["answer_multi_choice"]  # not tensor, list.

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    output = self.model(image, question)  # [batch_size, ans_vocab_size=1000]
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                ans_unk_idx = -1 # todo: tempcode
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
                            self.data_loader[phase].batch_step_size,
                            loss.item(),
                        )
                    )
            """
            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / self.data_loader[phase].batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(
                self.data_loader[phase].dataset
            )  # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(
                self.data_loader[phase].dataset
            )  # multiple choice
            """


    def run(
            self,
            learning_rate: float,
            step_size: int,
            gamma: float,
            save_step: int = 1,
            start_epochs: int = 0,
            num_epochs: int = 30,
            batch_size: int = 512,
    ):
        params = self.model.get_params()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        if start_epochs != 0:
            self.load_model()

        for epoch in range(start_epochs, num_epochs):
            self.step(
                epoch=epoch,
                num_epochs=num_epochs,
                batch_size=batch_size,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler
            )
            if (epoch + 1) % save_step == 0:
                self.save_model(epoch)
