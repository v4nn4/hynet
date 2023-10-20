from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import torch
from torch import nn


@dataclass
class TrainingReport:
    model_name: str
    model: nn.Module
    batch_size: int
    dataframe: pd.DataFrame

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        self.dataframe[["epoch", "train_loss", "test_loss"]].set_index("epoch").plot(
            title="Loss", ax=axes[0]
        )
        self.dataframe[["epoch", "train_accuracy", "test_accuracy"]].set_index(
            "epoch"
        ).plot(title="Accuracy", ax=axes[1])
        return fig, axes

    def save_fig(self, path) -> None:
        style.use("bmh")
        self.plot()
        plt.savefig(path)
        plt.close()

    def to_csv(self, path) -> None:
        self.dataframe.to_csv(path)

    def save_model(self, path) -> None:
        torch.save(self.model, path)
