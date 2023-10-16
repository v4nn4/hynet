import logging
from typing import Dict, Tuple

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
    learning_rate: float,
) -> TrainingReport:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    nb_samples_train = len(train_dataloader) * batch_size

    records = []
    best_accuracy, counter, early_stopping_patience = 0, 0, 5
    for epoch in range(nb_epochs):
        running_training_loss = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(batch_size))
            loss.backward()
            optimizer.step()
            running_training_loss += loss.item()
        training_loss = running_training_loss / nb_samples_train
        scheduler.step()

        # Compute metrics on test set
        test_loss, test_accuracy, test_accuracy_per_class = generate_test_metrics(
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
            f"Epoch = {epoch} / {nb_epochs}, Training Loss = {training_loss:.2f}, Test Loss = {test_loss:.2f}, Test Accuracy = {100 * test_accuracy:.2f}%"  # noqa: E501
        )
        record = {
            "epoch": epoch,
            "train_loss": training_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        for k, v in test_accuracy_per_class.items():
            record[f"test_accuracy_{k}"] = v
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return TrainingReport(
        model_name=model_name, model=model, batch_size=batch_size, dataframe=df
    )


def generate_test_metrics(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    test_dataloader: DataLoader,
    nb_classes: int,
    batch_size: int,
) -> Tuple[float, float, Dict[int, float]]:
    nb_samples_test = len(test_dataloader) * batch_size
    running_test_loss, nb_correct_predictions = 0, 0
    nb_correct_predictions_per_class = {c: 0 for c in range(nb_classes)}
    nb_samples_per_class = {c: 0 for c in range(nb_classes)}
    for inputs, labels in test_dataloader:
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
    return test_loss, test_accuracy, class_level_accuracy
