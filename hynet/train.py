import logging

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_exact_match

from .model import LeNet
from .report import TrainingReport

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


def train(
    experiment_name: str,
    N: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_classes: int,
    batch_size: int = 16,
    nb_epochs: int = 10,
    init_learning_rate: float = 0.1,
    steplr_gamma: float = 0.5,
    steplr_step_size: int = 10,
) -> TrainingReport:
    log_dir = f"build/runs/experiment_{experiment_name}"
    writer = SummaryWriter(log_dir=log_dir)

    model_name = "LeNet-5"
    mean, std = (
        2 / 3,
        np.sqrt(2) / 3,
    )  # mean and variance of a black square character
    mean, std = 0, 0
    for inputs, _ in train_dataloader:
        mean += inputs.mean().item()
        std += inputs.std().item()
    mean /= len(train_dataloader)
    std /= len(train_dataloader)
    logging.info(f"Using mean = {mean:.4f} and std = {std:.4f}")
    model = LeNet(N=N, num_classes=int(num_classes), mean=mean, std=std)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = StepLR(optimizer, step_size=steplr_step_size, gamma=steplr_gamma)

    # Train loop
    for epoch in np.arange(1, nb_epochs + 1):
        train_loss = 0
        train_acc = 0
        for inputs, labels in train_dataloader:
            logits = model(inputs)
            loss = criterion(logits, labels)

            train_loss += loss.item()
            max_indices = torch.argmax(logits, dim=1)
            preds = F.one_hot(max_indices, num_classes=logits.size(1)).float()
            train_acc += multiclass_exact_match(
                preds=preds, target=labels, num_classes=num_classes
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # Parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

        # Validation loop
        val_loss, val_acc = 0.0, 0.0
        with torch.inference_mode():
            for inputs, labels in val_dataloader:
                logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                max_indices = torch.argmax(logits, dim=1)
                preds = F.one_hot(max_indices, num_classes=logits.size(1)).float()
                val_acc += multiclass_exact_match(
                    preds=preds, target=labels, num_classes=num_classes
                )

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss},
            global_step=epoch,
        )
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc},
            global_step=epoch,
        )

        scheduler.step()
        logging.info(
            f"Epoch [{epoch}/{nb_epochs}] | Loss = {train_loss:.4f} | Accuracy (train) = {train_acc:.4f} | Accuracy (val) = {val_acc:.4f}"
        )

        if train_acc > 0.999:
            break

    writer.flush()
    writer.close()

    return TrainingReport(
        model_name=model_name, model=model, batch_size=batch_size, log_dir=log_dir
    )
