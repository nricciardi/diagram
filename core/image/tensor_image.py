from torchvision.io import read_image
import cv2
from core.image.image import Image
from dataclasses import dataclass
from torch import Tensor
import torch
from typing import Self
from src import DEVICE


@dataclass
class TensorImage(Image):

    tensor: Tensor

    @classmethod
    def from_str(cls, path: str) -> Self:
        if (path.lower().endswith('.png') or path.lower().endswith('.jpeg') or path.lower().endswith('.jpg')):
            tensor = read_image(path)
            if tensor.shape[0] == 4:
                tensor = tensor[:3, :, :]
            return TensorImage(tensor)
        if (path.lower().endswith('.bmp')):
            np_img = cv2.imread(path)
            if np_img.shape[2] == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2BGR)
            assert np_img.shape[2] == 3, f"Expected 3D numpy array, but got {np_img.ndim}D array for image {path}"
            tensor = torch.from_numpy(np_img).permute(2, 0, 1)
            return TensorImage(tensor)
            

    def as_tensor(self) -> Tensor:
        return self.tensor

    def to_device(self, device: str) -> None:
        self.tensor = self.tensor.to(device)
