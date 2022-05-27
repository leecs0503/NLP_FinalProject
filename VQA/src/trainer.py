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
from torch.nn.functional import l1_loss
from typing import List
from warmup_scheduler import GradualWarmupScheduler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


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

import torch
from torchvision.ops.boxes import box_convert, box_area

def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def _box_inter_union(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union

def generalized_iou_loss(gt_bboxes, pr_bboxes, is_debug=False):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / (union+1e-7)
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])

    assert (lt < rb).all()

    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    assert (enclosure > 1e-9).all()

    giou = iou - (enclosure-union)/(enclosure+1e-7)
    loss = 1. - giou

    if is_debug:
        print("lt")
        print(lt)
        print("rb")
        print(rb)
        print("wh")
        print(wh)
        print("inter")
        print(inter)
        print("iou")
        print(iou)
        print("enclosure")
        print(enclosure)
        print("giou")
        print(giou)
        print("loss")
        print(loss)


    return loss.sum(), iou



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
        print(load_path)
        states = torch.load(load_path)
        self.model.load_state_dict(states["model_state_dict"])
        return states["optimizer_state_dict"]

    def save_model(
        self,
        epoch: int,
        optimizer,
    ):
        save_path = os.path.join(
            self.model_path, f"{self.model.get_name()}-epoch-{epoch:02}.ckpt"
        )
        torch.save(
            {"epoch": epoch + 1, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path
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
        msg = "| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}, coor_exp: {:.4f}".format(
            phase.upper(),
            epoch + 1,
            num_epochs,
            batch_idx,
            int(
                len(self.data_loader[phase].dataset)
                / self.data_loader[phase].batch_size
            ),
            loss.item(),
            corr_exp / batch_size
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
        image_tensor_dict,
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
            if "vg" in phase:
                continue

            if "train" in phase:
                optimizer.step()
                scheduler.step(epoch)
                self.model.train()
            else:
                self.model.eval()

            if "vqa" in phase:
                for batch_idx, batch_sample in enumerate(self.data_loader[phase]):
                    # if batch_idx > 300:
                    #     if epoch <= 10:
                    #         break
                    # todo: 아래 로직을 함수로 빼기
                    image = batch_sample["image"].to(self.device)
                    question = batch_sample["question"].to(self.device)
                    question_token = batch_sample["question_token"]
                    label = batch_sample["answer_label"].to(self.device)
                    multi_choice = batch_sample["answer_multi_choice"]  # not tensor, list.

                    batch_step_size += 1

                    optimizer.zero_grad()

                    with torch.set_grad_enabled("train" in phase):
                        vqa_out, vg_out = self.model(
                            image, question
                        )  # [batch_size, ans_vocab_size=1000]
                        _, pred_exp = torch.max(vqa_out, 1)  # [batch_size]
                        loss = criterion(vqa_out, label)

                        if "train" in phase:
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
                # if self.prev_acc > epoch_acc_exp:
                #     early_stop = True
                # else:
                #     early_stop = False
                # self.prev_acc = epoch_acc_exp
                # return early_stop
            if "vg" in phase:
                for batch_idx, batch_sample in enumerate(self.data_loader[phase]):
                    # todo: 아래 로직을 함수로 빼기
                    image = batch_sample["image"].to(self.device)
                    question_token = batch_sample["question_token"]
                    bbox = batch_sample["bbox"].to(self.device)
                    bbox = box_convert(bbox, "xywh", "xyxy")

                    batch_step_size += 1

                    optimizer.zero_grad()

                    with torch.set_grad_enabled("train" in phase):
                        vqa_out, vg_out = self.model(
                            image, question_token
                        )  # [batch_size, ans_vocab_size=1000]
                        vg_out = box_convert(vg_out, "xywh", "xyxy")
                        
                        giou_loss, iou = generalized_iou_loss(vg_out, bbox)
                        l1_loss_va = l1_loss(vg_out,bbox,reduction='sum') / 224
                        loss = l1_loss_va  + giou_loss / 2
                        
                        if loss.item() / bbox.shape[0] > 100 or loss.item() < 0:
                            print(l1_loss_va, giou_loss / 2)
                            print(vg_out)
                            print(bbox)
                            generalized_iou_loss(vg_out, bbox, True)
                            exit(0)

                        if "train" in phase:
                            loss.backward()
                            optimizer.step()

                    # pred_exp[pred_exp == ans_unk_idx] = -9999
                    
                    batch_size = bbox.shape[0]
                    running_loss += loss.item() / batch_size
                    corr_exp = (iou > 0.5).sum()
                    running_corr_exp += corr_exp
                    running_count += batch_size
                    self.log_batch(
                        loss=loss / batch_size,
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


    def run(
        self,
        image_tensor_dict,
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
        '''
        CosineAnnealingWarmuoRestarts 
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
        '''
        #scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=50, gamma=0.5)

        if start_epochs != 0:
            optimizer.load_state_dict(self.load_model(start_epochs))

        for epoch in range(start_epochs, num_epochs):
            early_stop = self.step(
                image_tensor_dict,
                epoch=epoch,
                num_epochs=num_epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if (epoch + 1) % save_step == 0:
                self.save_model(epoch, optimizer)
            if early_stop == True:
                break
