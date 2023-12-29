import logging
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_exact_match
from PIL import ImageFont

from .model import LeNet
from .prepare import generate_classes, generate_dataset


def single_evaluate(image: torch.tensor, N: int, experiment_name: str) -> str:
    classes = generate_classes()
    nb_classes = len(classes)

    model = LeNet(N=N, num_classes=nb_classes)
    model.load_state_dict(torch.load(f"build/report/{experiment_name}/model.pt"))
    with torch.inference_mode():
        inputs = image.view(1, N, N)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted]


def evaluate(N: int, experiment_name: str, font_file_paths: List[str]) -> None:
    batch_size = 16
    classes = generate_classes()
    num_classes = len(classes)

    model = LeNet(N=N, num_classes=num_classes)
    model.load_state_dict(
        torch.load(f"build/report/{experiment_name}/model_weights.pt")
    )

    records = []
    logging.disable(logging.CRITICAL + 1)
    for font in font_file_paths:
        dataset = generate_dataset(N=N, font_file_paths=[font])
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        if len(dataloader) == 0:
            continue
        with torch.inference_mode():
            train_acc = 0
            for inputs, labels in dataloader:
                logits = model(inputs)
                max_indices = torch.argmax(logits, dim=1)
                preds = F.one_hot(max_indices, num_classes=logits.size(1)).float()
                train_acc += multiclass_exact_match(
                    preds=preds, target=labels, num_classes=num_classes
                )
            train_acc /= len(dataloader)
            name, style = ImageFont.truetype(f"{font}", N).getname()
            records.append(
                {
                    "accuracy": train_acc.item(),
                    "path": font,
                    "name": name,
                    "style": style,
                }
            )
    print(
        pd.DataFrame.from_records(records).sort_values(by="accuracy", ascending=False)
    )
