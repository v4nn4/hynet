from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn


@dataclass
class TrainingReport:
    model_name: str
    model: nn.Module
    batch_size: int
    dataframe: pd.DataFrame

    def to_csv(self, path) -> None:
        self.dataframe.to_csv(path)

    def save_model(self, path) -> None:
        torch.save(self.model, path)

    def to_gif(self, path) -> None:
        style.use("bmh")
        if not path.endswith(".gif"):
            raise ValueError("File path must end with .gif")
        nb_epochs = np.max(self.dataframe.index)
        images = []
        for epoch in range(nb_epochs):
            buffer = BytesIO()
            ax = (
                self.dataframe[["epoch", "train_loss", "test_loss", "test_accuracy"]]
                .set_index("epoch")
                .loc[:epoch]
                .plot(
                    title=f"Training Report [{self.model_name} (batch_size={self.batch_size})]"  # noqa: E501
                )
            )
            ax.set_ylim([0, 1])
            ax.set_xlim([0, nb_epochs])
            plt.legend(loc="lower right")
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image = Image.open(buffer)
            plt.close()
            images.append(image)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=100,
        )
