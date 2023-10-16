import os
import pickle

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    N, batch_size = 42, 16
    path = R"../build/datasets/hynet"

    test_dataset = pickle.load(open(os.path.join(path, "test_dataset.pkl"), "rb"))
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    nb_classes = len(list(set([c for (_, c) in test_dataset])))
    nb_samples_test = len(test_dataloader) * batch_size

    model = torch.load(f"../build/logs/train/model_{batch_size}.pth")
    with torch.no_grad():
        nb_correct_predictions = 0
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            nb_correct_predictions += (predicted == labels.flatten()).sum().item()
        test_accuracy = nb_correct_predictions / nb_samples_test
        print(test_accuracy)
