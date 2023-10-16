import itertools
import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import TensorDataset

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)
BLACK = 0
WHITE = 255

# Model parameters
NB_ROTATIONS = 5
NB_BLUR_RADIUSES = 5


def generate_classes(characters_to_ignore: str = "") -> List[str]:
    # Armenian alphabet is located between 0x531 and 0x58A
    characters = [chr(i) for i in range(0x531, 0x58A)]
    characters_to_ignore = "՗՘ՙ՚՛՜՝՞՟ՠֈ։և"
    characters = [c for c in characters if c not in list(characters_to_ignore)]
    return characters


def generate_character_image(
    character: str,
    font: ImageFont.FreeTypeFont,
    N,
    rotation: float = 0.0,
    blur_radius: float = 0.0,
    mode_filter: int = 0,
) -> torch.Tensor:
    image = Image.new("L", size=(N, N), color=WHITE)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = font.getbbox(character)
    draw.text(
        (-x0 + (N - (x1 - x0)) / 2, -y0 + (N - (y1 - y0)) / 2),
        character,
        font=font,
        fill=BLACK,
    )
    image = image.rotate(rotation, fillcolor=WHITE)
    image = image.filter(ImageFilter.ModeFilter(mode_filter))
    image = image.filter(ImageFilter.BoxBlur(blur_radius))
    pixel_values = list(image.getdata())
    T = torch.tensor(pixel_values, dtype=torch.float32) / 255.0
    T = T.view((1, N, N))
    return T


def generate_dataset(
    N,
    font_names: List[str] = ["arial"],
    rotation_angle: float = 10,
    split_ratio: float = 0.8,
) -> Tuple[TensorDataset, TensorDataset]:
    """Generate train and test datasets"""

    # Generate classes
    classes = generate_classes()
    logging.info(f"Characters used     : {''.join(classes)}")
    logging.info(f"Number of classes   : {len(classes)}")

    # Pre-processing
    rotations = np.arange(
        -rotation_angle, 2 * rotation_angle, 3 * rotation_angle / NB_ROTATIONS
    )
    blur_radiuses = np.arange(0, 3.0, 3.0 / NB_BLUR_RADIUSES)
    mode_filters = np.array([0, 2, 4])
    logging.info(
        (
            f"Data augmentation   : {len(font_names)} fonts, "
            f"{len(rotations)} rotations, {len(blur_radiuses)} blur radiuses, "
            f"{len(mode_filters)} mode filters"
        )
    )
    nb_samples = (
        len(classes)
        * len(font_names)
        * len(rotations)
        * len(blur_radiuses)
        * len(mode_filters)
    )
    logging.info(f"Number of samples   : {nb_samples}")

    # Initialize input and label tensors, then fill
    input_tensor = torch.zeros((nb_samples, 1, N, N), dtype=torch.float32)
    label_tensor = torch.zeros((nb_samples, 1), dtype=torch.uint8)
    for font_name in font_names:
        font = ImageFont.truetype(f"{font_name}.ttf", N)
        for i, (character, rotation, blur_radius, mode_filter) in enumerate(
            itertools.product(classes, rotations, blur_radiuses, mode_filters)
        ):
            T = generate_character_image(
                character, font, N, rotation, blur_radius, mode_filter
            )
            input_tensor[i, :] = T
            label_tensor[i] = classes.index(character)

    # Split between train and test datasets
    split_index = int(split_ratio * nb_samples)
    shuffled_indices = torch.randperm(input_tensor.size(0))
    input_tensor = input_tensor[shuffled_indices]
    label_tensor = label_tensor[shuffled_indices]

    train_dataset = TensorDataset(
        input_tensor[:split_index], label_tensor[:split_index]
    )
    test_dataset = TensorDataset(input_tensor[split_index:], label_tensor[split_index:])
    return train_dataset, test_dataset
