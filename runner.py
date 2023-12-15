import os
import pickle

import fire
import torch
from torch.utils.data import DataLoader

from hynet.model import LeNet
from hynet.prepare import generate_classes, generate_dataset
from hynet.train import train


class Runner(object):
    def prepare(self, N: int = 56):
        """Train model

        Args:
            N (int, optional): image size. Defaults to 56.
        """
        N = 56  # 56x56 pixels
        font_names = ["hynet/fonts/Mk_Parz_U-Italic"]
        train_dataset, test_dataset = generate_dataset(
            font_names=font_names, N=N, split_ratio=0.8
        )

        path = R"build/datasets/hynet"
        os.makedirs(path, exist_ok=True)
        train_dataset = pickle.load(open(os.path.join(path, "train_dataset.pkl"), "rb"))
        test_dataset = pickle.load(open(os.path.join(path, "test_dataset.pkl"), "rb"))
        with open(os.path.join(path, "train_dataset.pkl"), "wb") as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(path, "test_dataset.pkl"), "wb") as f:
            pickle.dump(test_dataset, f)

    def train(
        self,
        N: int = 56,
        batch_size: int = 16,
        nb_epochs: int = 10,
        learning_rate: float = 3.0e-4,
        steplr_step_size: int = 20,
        early_stopping_patience: int = 3,
    ):
        """Train model

        Args:
            N (int, optional): image size. Defaults to 56.
            batch_size (int, optional): batch size. Defaults to 16.
            nb_epochs (int, optional): number of epochs. Defaults to 10.
            learning_rate (float, optional): learning rate. Defaults to 3.0e-4.
            steplr_step_size: (int, optional): learning rate step size. Defaults to 20.
            early_stopping_patience: (int, optional): early stopping patience. Defaults to 3.
        """
        path = R"build/datasets/hynet"

        train_dataset = pickle.load(open(os.path.join(path, "train_dataset.pkl"), "rb"))
        train_data = next(iter(train_dataset))
        mean, std = train_data[0].mean(), train_data[0].std()
        test_dataset = pickle.load(open(os.path.join(path, "test_dataset.pkl"), "rb"))
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        # Initialize model
        nb_classes = len(generate_classes())
        model_name = "LeNet-5"
        model = LeNet(N=N, C=int(nb_classes), mean=mean, std=std)

        # Train
        report = train(
            model_name=model_name,
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            nb_classes=int(nb_classes),
            batch_size=batch_size,
            nb_epochs=nb_epochs,
            init_learning_rate=learning_rate,
            steplr_step_size=steplr_step_size,
            early_stopping_patience=early_stopping_patience,
        )
        train_folder = "build/logs/train"
        os.makedirs(train_folder, exist_ok=True)
        report.save_model(os.path.join(train_folder, "model.pt"))  # save model
        report.save_fig(os.path.join(train_folder, "report.svg"))  # plot report as png
        report.to_csv(os.path.join(train_folder, "report.csv"))  # export full report

    def evaluate(self, batch_size: int = 16):
        """Run inference on trained model

        Args:
            batch_size (int, optional): batch size. Defaults to 16.
        """
        path = R"../build/datasets/hynet"

        test_dataset = pickle.load(open(os.path.join(path, "test_dataset.pkl"), "rb"))
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        # nb_classes = len(list(set([c for (_, c) in test_dataset])))
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


if __name__ == "__main__":
    fire.Fire(Runner)
