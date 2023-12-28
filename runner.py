import datetime
import os
import pickle

import fire
import numpy as np
from torch.utils.data import DataLoader

from hynet.model import LeNet
from hynet.prepare import generate_classes, generate_dataset
from hynet.train import train
from hynet.evaluate import evaluate


def find_fonts(font_prefix: str):
    fonts = []
    for _, _, files in os.walk("hynet/fonts"):
        for file in files:
            if file.startswith(font_prefix):
                fonts.append(os.path.join("hynet/fonts", file))
    return fonts


class Runner(object):
    def prepare(self, N: int = 56):
        """Train model

        Args:
            N (int, optional): image size. Defaults to 56.
        """
        train_fonts = [
            "DejaVuSans",
            "Mardoto",
        ]
        val_fonts = ["Mk_Parz_U"]

        train_font_paths = []
        for font in train_fonts:
            train_font_paths += find_fonts(font)
        val_font_paths = []
        for font in val_fonts:
            val_font_paths += find_fonts(font)
        train_dataset = generate_dataset(N=N, font_file_paths=train_font_paths)
        val_dataset = generate_dataset(N=N, font_file_paths=val_font_paths)

        path = R"build/datasets/hynet"
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "train_dataset.pkl"), "wb") as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(path, "val_dataset.pkl"), "wb") as f:
            pickle.dump(val_dataset, f)

    def train(
        self,
        N: int = 56,
        batch_size: int = 16,
        nb_epochs: int = 10,
        learning_rate: float = 3.0e-4,
        steplr_step_size: int = 20,
    ):
        """Train model

        Args:
            N (int, optional): image size. Defaults to 56.
            batch_size (int, optional): batch size. Defaults to 16.
            nb_epochs (int, optional): number of epochs. Defaults to 10.
            learning_rate (float, optional): learning rate. Defaults to 3.0e-4.
            steplr_step_size: (int, optional): learning rate step size. Defaults to 20.
        """
        path = R"build/datasets/hynet"

        train_dataset = pickle.load(open(os.path.join(path, "train_dataset.pkl"), "rb"))
        val_dataset = pickle.load(open(os.path.join(path, "val_dataset.pkl"), "rb"))
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        # Initialize model
        nb_classes = len(generate_classes())
        model_name = "LeNet-5"
        mean, std = (
            2 / 3,
            np.sqrt(2) / 3,
        )  # mean and variance of a black square character
        model = LeNet(N=N, num_classes=int(nb_classes), mean=mean, std=std)

        # Train
        today = datetime.datetime.today()
        experiment_name = today.strftime("%Y-%m-%d_%H:%M:%S")
        report = train(
            experiment_name=experiment_name,
            model_name=model_name,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            nb_classes=int(nb_classes),
            batch_size=batch_size,
            nb_epochs=nb_epochs,
            init_learning_rate=learning_rate,
            steplr_step_size=steplr_step_size,
        )
        report_folder = f"build/report/{experiment_name}"
        os.makedirs(report_folder, exist_ok=True)
        report.save_model(os.path.join(report_folder, "model.pt"))  # save model
        report.to_csv(os.path.join(report_folder, "report.csv"))  # export full report
        report.save_fig(os.path.join(report_folder, "report.svg"))  # plot report as png

    def evaluate(self, N: int = 56, batch_size: int = 16):
        """Run inference on trained model

        Args:
            N (int, optional): image size. Defaults to 56.
            batch_size (int, optional): batch size. Defaults to 16.
        """
        evaluate(N=N, batch_size=batch_size)


if __name__ == "__main__":
    fire.Fire(Runner)
