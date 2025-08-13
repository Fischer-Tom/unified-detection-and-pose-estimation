from typing import Union
from PIL import Image

from enum import Enum, auto

import torch
import numpy as np
import torchvision.transforms.functional as tF


class ImageVariant(Enum):
    """Enum for image variants."""

    TORCH = auto()
    PIL = auto()
    NUMPY = auto()


ImageType = Union[torch.Tensor, Image.Image, np.ndarray]
""" The type of an image. """


def convert_tensor_to(
    image_tensor: "torch.Tensor", output_type: "ImageVariant"
) -> "ImageType":
    if output_type == ImageVariant.TORCH:
        return image_tensor
    elif output_type == ImageVariant.PIL:
        return tF.to_pil_image(image_tensor)
    elif output_type == ImageVariant.NUMPY:
        return np.array(tF.to_pil_image(image_tensor))
    else:
        raise ValueError("Invalid output type")


def convert_pil_to(image_pil, output_type: "ImageVariant") -> "ImageType":
    if output_type == ImageVariant.TORCH:
        return tF.to_tensor(image_pil)
    elif output_type == ImageVariant.PIL:
        return image_pil
    elif output_type == ImageVariant.NUMPY:
        return np.array(image_pil)
    else:
        raise ValueError("Invalid output type")


def convert_numpy_to(image_numpy, output_type: "ImageVariant") -> "ImageType":
    if output_type == ImageVariant.TORCH:
        return tF.to_tensor(image_numpy)
    elif output_type == ImageVariant.PIL:
        if image_numpy.dtype == np.float32:
            image_numpy = (image_numpy * 255).astype(np.uint8)
        return Image.fromarray(image_numpy)
    elif output_type == ImageVariant.NUMPY:
        return image_numpy
    else:
        raise ValueError("Invalid output type")


def convert_to(image: "ImageType", output_type: "ImageVariant") -> "ImageType":
    """
    Converts an image to a different type.
    Args:
        image: The input image.
        output_type: The output type of the image.
    Returns:
        ImageType: The converted image.
    """
    if isinstance(image, torch.Tensor):
        return convert_tensor_to(denormalize(image), output_type)
    elif isinstance(image, Image.Image):
        return convert_pil_to(image, output_type)
    elif isinstance(image, np.ndarray):
        return convert_numpy_to(image, output_type)
    else:
        raise ValueError("Invalid input type")


def denormalize(
    image_tensor: "torch.Tensor",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    output_type: "ImageVariant" = ImageVariant.TORCH,
) -> "ImageType":
    """
    Denormalizes an image tensor.
    Args:
        image_tensor: The input image tensor.
        mean: The mean used for normalization.
        std: The standard deviation used for normalization.
        output_type: The output type of the image.
    Returns:
        ImageType: The denormalized image.
    """
    mean = torch.as_tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)[
        ..., None, None
    ]
    std = torch.as_tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)[
        ..., None, None
    ]
    image_tensor = image_tensor * std + mean
    return convert_tensor_to(image_tensor, output_type)