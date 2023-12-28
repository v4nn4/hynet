import os
import pickle

import torch
from torch.utils.data import DataLoader

from .model import LeNet
from .prepare import generate_classes


def single_evaluate(image: torch.tensor, N: int) -> str:
    classes = generate_classes()
    nb_classes = len(classes)

    model = LeNet(N=N, num_classes=nb_classes)
    model.load_state_dict(torch.load("build/logs/train/model.pt"))
    model.eval()
    with torch.no_grad():
        inputs = image.view(1, N, N)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted]


def evaluate(N: int = 56, batch_size: int = 16) -> None:
    path = R"build/datasets/hynet"

    test_dataset = pickle.load(open(os.path.join(path, "test_dataset.pkl"), "rb"))
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    classes = generate_classes()
    nb_classes = len(classes)
    nb_samples_test = len(test_dataloader) * batch_size

    model = LeNet(N=N, num_classes=nb_classes)
    model.load_state_dict(torch.load("build/logs/train/model.pt"))
    model.eval()
    with torch.no_grad():
        nb_correct_predictions = 0
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            nb_correct_predictions += (predicted == labels.flatten()).sum().item()
        test_accuracy = nb_correct_predictions / nb_samples_test
        print(test_accuracy)
