from torchvision.io import read_image
import cv2

from core.image.image import Image
from dataclasses import dataclass
from torch import Tensor
import torch


@dataclass
class TensorImage(Image):
    tensor: Tensor

    @staticmethod
    def from_str(path: str) -> 'Image':
        if (path.lower().endswith('.png') or path.lower().endswith('.jpeg')):
            return TensorImage(read_image(path))
        if (path.lower().endswith('.bmp')):
            np_img = cv2.imread(path)
            tensor = torch.from_numpy(np_img).permute(2, 0, 1)
            return TensorImage(tensor)
            

    def as_tensor(self) -> Tensor:
        return self.tensor
