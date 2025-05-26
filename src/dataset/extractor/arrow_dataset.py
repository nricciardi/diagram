import json
import os
from enum import Enum
from typing import Tuple, Dict, List
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.extractor.arrow.dataset_generator import ARROW_CATEGORY


class ArrowType(Enum):
    HEAD = "head"
    TAIL = "tail"



class ArrowDataset(Dataset):
    """
    Dataset class for fine-tuning the object detection network
    """

    def __init__(self, arrow_type: str, info_file: str, image_dir: str, patch_size: int):
        """

        Args:
            arrow_type: 'head' or 'tail'
            info_file:
            image_dir:
            patch_size:
        """

        assert(arrow_type == ArrowType.HEAD.value or arrow_type == ArrowType.TAIL.value)

        self.arrow_type = arrow_type
        self.patch_size = patch_size
        self.index_lookup: list = []
        with open(info_file, 'r') as f:

            self.info = json.load(f)
            for i, annotation in enumerate(self.info["annotations"]):
                if annotation["category_id"] == ARROW_CATEGORY:
                    self.index_lookup.append(i)

        self.image_dir = image_dir

    def __len__(self):
        return len(self.index_lookup)


    def __process(self, image: torch.tensor, center_x: float, center_y: float):

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


    def __getitem__(self, idx: int) -> Tuple[torch.tensor, Tuple[int, int], Tuple[int, int]]:

        annotation = self.info["annotations"][idx]

        head_x: float = annotation["keypoints"][0]
        head_y: float = annotation["keypoints"][1]
        tail_x: float = annotation["keypoints"][3]
        tail_y: float = annotation["keypoints"][4]

        head_x: int = int(head_x)
        head_y: int = int(head_y)
        tail_x: int = int(tail_x)
        tail_y: int = int(tail_y)

        image_id = annotation["image_id"]
        image_info = self.info["images"][image_id]

        # Load image from file (grayscale assumed)
        image = cv2.imread(image_info["file_name"], cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image).unsqueeze(0)

        if self.arrow_type == ArrowType.HEAD:
            image = self.__process(image, head_x, head_y)
        elif self.arrow_type == ArrowType.TAIL:
            image = self.__process(image, tail_x, tail_y)
        else:
            raise ValueError("Neither head or tail arrow type found")

        return image, (head_x, head_y), (tail_x, tail_y)


