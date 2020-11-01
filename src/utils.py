from PIL import Image, ImageFilter  # type: ignore
import torch
from typing import List


def stringify_digits(t: torch.Tensor) -> str:
    return "".join([str(n) for n in t.numpy()])


def detect_edges(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.FIND_EDGES)


def target_transform(folder_names: List[str]):
    def t(i: int) -> torch.Tensor:
        return torch.tensor([int(n) for n in folder_names[i]], dtype=torch.long)

    return t
