import torch

from core.image.image import Image
from dataclasses import dataclass
from torch import Tensor
from typing import List


@dataclass
class ListImage(Image):
    content: List[float]

    def as_tensor(self) -> Tensor:
        return torch.FloatTensor(self.content)

