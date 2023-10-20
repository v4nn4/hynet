import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .report import TrainingReport

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


def train(
    model_name: str,
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    nb_classes: int,
    batch_size: int,
    nb_epochs: int,
    init_learning_rate: float = 0.1,
    steplr_gamma: float = 0.5,
    steplr_step_size: int = 10,
    early_stopping_patience: int = 5,
) -> TrainingReport:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = StepLR(optimizer, step_size=steplr_step_size, gamma=steplr_gamma)

    records = []
    best_accuracy, counter = 0, 0
    for epoch in np.arange(1, nb_epochs + 1):
        running_training_loss = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(batch_size))
            loss.backward()
            optimizer.step()
            running_training_loss += loss.item()
        scheduler.step()

        # Compute metrics on training set
        (
            train_loss,
            train_accuracy,
            training_accuracy_per_class,
        ) = generate_metrics(model, criterion, train_dataloader, nb_classes, batch_size)

        # Compute metrics on test set
        test_loss, test_accuracy, test_accuracy_per_class = generate_metrics(
            model, criterion, test_dataloader, nb_classes, batch_size
        )

        if test_accuracy <= best_accuracy:
            counter += 1
        if counter > early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        if test_accuracy > best_accuracy:
            counter = 0
        best_accuracy = test_accuracy

        # Log progress and append report
        logging.info(
            f"Epoch = {epoch} / {nb_epochs}, Training Loss = {train_loss:.2f}, Test Loss = {test_loss:.2f}, Test Accuracy = {100 * test_accuracy:.2f}%"  # noqa: E501
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        for k, v in test_accuracy_per_class.items():
            record[f"test_accuracy_{k}"] = v
        for k, v in training_accuracy_per_class.items():
            record[f"test_accuracy_{k}"] = v
        records.append(record)

    df = pd.DataFrame.from_records(records)

    return TrainingReport(
        model_name=model_name, model=model, batch_size=batch_size, dataframe=df
    )


def generate_metrics(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    data_loader: DataLoader,
    nb_classes: int,
    batch_size: int,
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()  # eval mode to remove dropout
    nb_samples_test = len(data_loader) * batch_size
    running_test_loss, nb_correct_predictions = 0, 0
    nb_correct_predictions_per_class = {c: 0 for c in range(nb_classes)}
    nb_samples_per_class = {c: 0 for c in range(nb_classes)}
    for inputs, labels in data_loader:
        outputs = model(inputs)
        test_loss = criterion(outputs, labels.view(batch_size)).item()
        running_test_loss += test_loss
        _, predicted = torch.max(outputs, 1)
        nb_correct_predictions += (predicted == labels.flatten()).sum().item()
        for prediction, label in zip(predicted, labels):
            if prediction == label:
                nb_correct_predictions_per_class[label.item()] += 1
            nb_samples_per_class[label.item()] += 1
    test_loss = running_test_loss / nb_samples_test
    test_accuracy = nb_correct_predictions / nb_samples_test
    class_level_accuracy = {
        c: nb_correct_predictions_per_class[c] / nb_samples_per_class[c]
        if nb_samples_per_class[c] > 0
        else torch.nan
        for c in range(nb_classes)
    }
    model.train()
    return test_loss, test_accuracy, class_level_accuracy
