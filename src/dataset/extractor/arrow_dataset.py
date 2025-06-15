import json
import os
from enum import Enum
from typing import Tuple, Dict, List
import cv2
import torch
import torch.nn.functional as F
from numpy.ma.core import anomalies
from torch.utils.data import Dataset
import random

import matplotlib.pyplot as plt
from src.extractor.arrow.dataset_generator import ARROW_CATEGORY


class ArrowType(Enum):
    HEAD = "head"
    TAIL = "tail"



class ArrowDataset(Dataset):
    """
    Dataset class for fine-tuning the object detection network
    """

    def __init__(self, arrow_type: str, info_file: str, image_dir: str, patch_size: int, output_image_size: int, sigma: float = 2):
        """

        Args:
            arrow_type: 'head' or 'tail'
            info_file:
            image_dir:
            patch_size:
        """

        assert(arrow_type == ArrowType.HEAD.value or arrow_type == ArrowType.TAIL.value)
        assert(patch_size <= output_image_size)

        self.output_image_size = output_image_size
        self.sigma = sigma
        self.arrow_type = arrow_type
        self.patch_size = patch_size
        with open(info_file, 'r') as f:

            self.info = [annotation for annotation in json.load(f) if annotation["label"] == arrow_type]

        self.image_dir = image_dir

    def __len__(self):
        return len(self.info)


    def __crop_patch(self, image: torch.tensor, center_x: float, center_y: float):

        C, H, W = image.shape
        half_size = self.patch_size // 2

        cx = int(round(center_x))
        cy = int(round(center_y))

        x1 = cx - half_size
        y1 = cy - half_size
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size

        # Padding if patch is greater than image itself
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

        patch = image[:, y1:y2, x1:x2]
        return patch

    def __compute_heatmap(self, center_x: int, center_y: int, size: int):

        x = torch.arange(size).float()
        y = x.view(-1, 1)

        heatmap = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * self.sigma ** 2))

        return heatmap


    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:

        annotation = self.info[idx]

        target_x: int = int(annotation["target_x"])
        target_y: int = int(annotation["target_y"])

        image_path = os.path.join(self.image_dir, annotation["image"])

        # Load image from file (grayscale assumed)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    # (H, W)
        image = torch.from_numpy(image).unsqueeze(0)

        crop_image = self.__crop_patch(image, target_x, target_y)    # (1, patch_size, patch_size)
        crop_image = crop_image.squeeze()

        center_x = random.randint(0, self.output_image_size - 1)
        center_y = random.randint(0, self.output_image_size - 1)

        label = self.__compute_heatmap(center_x, center_y, self.output_image_size)


        h, w = crop_image.shape

        margin_x = self.output_image_size - w
        margin_y = self.output_image_size - h

        center_x = random.randint(w // 2, margin_x + w // 2)
        center_y = random.randint(h // 2, margin_y + h // 2)

        top_left_x = center_x - w // 2
        top_left_y = center_y - h // 2

        image = torch.zeros(self.output_image_size, self.output_image_size, dtype=torch.uint8)
        image[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = crop_image

        return image, label


def overlay_grayscale_red_torch(base_img: torch.Tensor, red_overlay: torch.Tensor, alpha: float = 0.5):
    """
    base_img:     (H, W) grayscale image (float or uint8)
    red_overlay:  (H, W) grayscale to overlay in red
    alpha:        blending factor for the red overlay
    """
    # Assicurati che siano float32 tra 0 e 1
    if base_img.dtype != torch.float32:
        base_img = base_img.float() / 255.0
    if red_overlay.dtype != torch.float32:
        red_overlay = red_overlay.float() / 255.0

    # Crea immagine RGB: (H, W, 3)
    rgb = torch.stack([base_img, base_img, base_img], dim=-1)  # (H, W, 3)

    # Aggiungi overlay sul canale rosso (canale 0)
    rgb[:, :, 0] = torch.clamp(rgb[:, :, 0] + alpha * red_overlay, 0, 1)

    # Porta su CPU per plotting
    rgb_np = rgb.detach().cpu().numpy()

    # Visualizza
    plt.imshow(rgb_np)
    plt.axis("off")
    plt.title("Overlay Rosso su Grayscale (Torch)")
    plt.show()

if __name__ == '__main__':
    dataset = ArrowDataset(
        ArrowType.HEAD.value,
        "/home/nricciardi/Repositories/diagram/dataset/arrow/head/train.json",
        "/home/nricciardi/Repositories/diagram/dataset/arrow/head/train",
        64,
        128,
        2
    )

    image, label = dataset.__getitem__(0)

    overlay_grayscale_red_torch(image, label)

    plt.imshow(image)
    plt.show()

    plt.imshow(label)
    plt.show()

