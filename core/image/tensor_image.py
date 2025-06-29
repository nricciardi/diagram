from core.image.image import Image
from dataclasses import dataclass
from torch import Tensor
import torch
from typing import Self
from PIL import Image as PILImage
import numpy as np


@dataclass
class TensorImage(Image):

    tensor: Tensor

    @classmethod
    def from_str(cls, path: str) -> Self:

        pil_img = PILImage.open(path).convert("RGBA")
        np_img = np.array(pil_img)

        if np_img.shape[2] == 4:
            alpha = np_img[:, :, 3:] / 255.0
            rgb = np_img[:, :, :3].astype(np.float32)
            white_bg = 255.0 * (1 - alpha)
            rgb = rgb * alpha + white_bg
            np_img = rgb.astype(np.uint8)
        else:
            np_img = np_img[:, :, :3]


        if len(np_img.shape) == 2:
            np_img = np.stack([np_img] * 3, axis=-1)

        tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()  # C, H, W
        return TensorImage(tensor)
            

    def as_tensor(self) -> Tensor:
        return self.tensor

    def to_device(self, device: str) -> None:
        self.tensor = self.tensor.to(device)

    def get_device(self) -> str:
        return self.tensor.device.type
