import itertools
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import TensorDataset

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)
BLACK = 0
WHITE = 255

# Model parameters
NB_ROTATIONS = 4
NB_BLUR_RADIUSES = 4
NB_NOISE_INTENSITIES = 4


def generate_classes() -> List[str]:
    """Generate the list of characters, i.e. the number of classes"""
    # Armenian alphabet is located between 0x531 and 0x58A
    # Lower case characters start at 0x561
    characters = [chr(i) for i in range(0x561, 0x58A)]
    characters_to_ignore = "՗՘ՙ՚՛՜՝՞՟ՠֈ։և"
    characters = [c for c in characters if c not in list(characters_to_ignore)]
    return characters


def generate_character_image(
    character: str,
    font: ImageFont.FreeTypeFont,
    N: int,
    rotation: float = 0.0,
    blur_radius: float = 0.0,
    mode_filter: int = 0,
    noise_intensity: float = 0.0,
) -> torch.Tensor:
    """Generates a N x N pixels image of a character. Methodology :
      1. Generate a `PIL.Image` in L mode
      2. Draws the character using the input font
      3. Applies some `PIL.ImageFilter` transformations
      4. Convert to `torch.Tensor` with type `torch.float32` and values between 0 and 1
      5. Adds a channel dimension (black and white)

    Args:
        character: Character as a string
        font: font object
        N: number of pixels for each dimension (width, height)
        rotation: rotation angle
        blur_radius: box blur radius
        mode_filter: mode filter
        noise_intensity: intensity of noise. 1.0 means N(0, 1)
    """
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
    image_array = np.array(image)
    noise = np.random.normal(0, 10, image_array.shape).astype(np.uint8)
    pixel_values = np.clip(image_array - noise * noise_intensity, 0, 255).astype(
        np.uint8
    )
    T = torch.tensor(pixel_values, dtype=torch.float32) / 255.0
    T = T.view((1, N, N))
    return T


def generate_dataset(
    N: int,
    font_file_paths: List[str],
) -> Tuple[TensorDataset, TensorDataset]:
    """Generate train and test datasets

    Args:
        N: number of pixels for each dimension (width, height)
        font_names: list of font names. font.ttf must be available locally
    """

    # Generate classes
    classes = generate_classes()
    num_classes = len(classes)
    logging.info(f"Characters used     : {''.join(classes)}")
    logging.info(f"Number of classes   : {num_classes}")

    # Pre-processing
    rotations = np.arange(-10, 30, 40 / NB_ROTATIONS)
    blur_radiuses = np.arange(0, 3.0, 3.0 / NB_BLUR_RADIUSES)
    mode_filters = np.array([0, 2, 4])
    noise_intensities = np.arange(0, 1.0, 1.0 / NB_NOISE_INTENSITIES)
    logging.info(
        (
            f"Data augmentation   : {len(font_file_paths)} fonts, "
            f"{len(rotations)} rotations, {len(blur_radiuses)} blur radiuses, "
            f"{len(mode_filters)} mode filters, {len(noise_intensities)} noise intensities"
        )
    )
    nb_samples = (
        num_classes
        * len(font_file_paths)
        * len(rotations)
        * len(blur_radiuses)
        * len(mode_filters)
        * len(noise_intensities)
    )
    logging.info(f"Number of samples   : {nb_samples}")

    # Initialize input and label tensors, then fill
    input_tensor = torch.zeros((nb_samples, 1, N, N), dtype=torch.float32)
    label_index_tensor = torch.zeros((nb_samples), dtype=torch.int64)
    for font_file_path in font_file_paths:
        font = ImageFont.truetype(f"{font_file_path}", N)
        for i, (
            character,
            rotation,
            blur_radius,
            mode_filter,
            noise_intensity,
        ) in enumerate(
            itertools.product(
                classes, rotations, blur_radiuses, mode_filters, noise_intensities
            )
        ):
            T = generate_character_image(
                character, font, N, rotation, blur_radius, mode_filter, noise_intensity
            )
            input_tensor[i, :] = T
            index = classes.index(character)
            label_index_tensor[i] = index
    label_tensor = F.one_hot(label_index_tensor, num_classes=num_classes).float()

    # Shuffle
    shuffled_indices = torch.randperm(input_tensor.size(0))
    input_tensor = input_tensor[shuffled_indices]
    label_tensor = label_tensor[shuffled_indices]

    dataset = TensorDataset(input_tensor, label_tensor)
    return dataset
