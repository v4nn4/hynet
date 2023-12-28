from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.style as style
import torch
from tbparse import SummaryReader
from torch import nn


@dataclass
class TrainingReport:
    model_name: str
    model: nn.Module
    batch_size: int
    log_dir: str

    def plot(self):
        reader = SummaryReader(log_path=self.log_dir, extra_columns={"dir_name"})
        df = reader.get_events("scalars")
        df = df.rename(columns={"step": "epoch"})
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        df[df.tag == "Accuracy"].pivot_table(
            columns=["dir_name"], index="epoch", values="value"
        ).rename(
            columns={
                "Accuracy_train/acc": "train_accuracy",
                "Accuracy_val/acc": "val_accuracy",
            }
        ).plot(
            title="Accuracy", ax=axes[0]
        )
        df[df.tag == "Loss"].pivot_table(
            columns=["dir_name"], index="epoch", values="value"
        ).rename(
            columns={"Loss_train/loss": "train_loss", "Loss_val/loss": "val_loss"}
        ).plot(
            title="Loss", ax=axes[1]
        )
        axes[0].legend(["train_accuracy", "val_accuracy"])
        axes[1].legend(["train_loss", "val_loss"])
        return fig, axes

    def save_fig(self, path) -> None:
        style.use("bmh")
        self.plot()
        plt.savefig(path)
        plt.close()

    def to_csv(self, path: str) -> None:
        reader = SummaryReader(log_path=self.log_dir, extra_columns={"dir_name"})
        df = reader.get_events("scalars")
        df.to_csv(path)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
