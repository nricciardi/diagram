import json
import os
from typing import Tuple, Dict, List

from torch.utils.data import Dataset
from torchvision import tv_tensors as t
from torchvision.io import read_image, ImageReadMode
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class ObjectDetectionDataset(Dataset):
    """
    Dataset class for fine-tuning the object detection network

    :param annotations_file: path to the labels file
    :param img_dir: path to the images directory
    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, synthetic_bbox_size: int = 64):
        with open(annotations_file, 'r') as f:
            self.labels = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.synthetic_bbox_size = synthetic_bbox_size

    def __len__(self):
        return len(self.labels['images'])

    def generate_synthetic_bbox(self, image: torch.Tensor, center_x: int, center_y: int, image_id: int) -> List[int]:
        """
        Given an image, produce a synthetic bbox centered on position

        Returns:
            [x, y, width, height]
        """

        _, H, W = image.shape

        if center_x >= W or center_y >= H:
            raise ValueError(f"center_x: {center_x}, W: {W}, center_y: {center_y}, H: {H}, image_id: {image_id}")
        assert center_x < W
        assert center_y < H

        half_bbox = self.synthetic_bbox_size // 2

        x: int = max(0, center_x - half_bbox)
        y: int = max(0, center_y - half_bbox)

        x2: int = min(W, center_x + half_bbox)
        y2: int = min(H, center_y + half_bbox)

        width = x2 - x
        height = y2 - y

        return [x, y, width, height]

    def plot_image_with_keypoints_and_bbox(self, image_tensor, keypoints, bbox=None):
        """
        Plots an image tensor with keypoints and an optional bounding box.

        Args:
            image_tensor (torch.Tensor): Tensor of shape (C, H, W)
            keypoints (list or np.ndarray): List or array of (x, y) keypoints
            bbox (list of int, optional): [x, y, width, height]
        """
        keypoints = np.array(keypoints)

        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError("image_tensor must be a torch.Tensor")

        if image_np.shape[2] == 1:
            image_np = image_np.squeeze(-1)
            cmap = 'gray'
        else:
            cmap = None

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np, cmap=cmap)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=20, marker='x', label="Keypoints")

        # Draw bounding box if provided
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h,
                                     linewidth=2, edgecolor='lime', facecolor='none', label="BBox")
            ax.add_patch(rect)

        ax.set_title("Image with Keypoints and Bounding Box")
        ax.axis('off')
        plt.legend()
        plt.show(block=True)

    def __getitem__(self, idx) -> Tuple[t.Image, Dict]:
        """
        Get an item from the dataset

        :param idx: The index of the item to retrieve

        :return: A tuple containing the image tensor and a dictionary containing label and other stuff.


        See https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """
        images: List[Dict] = self.labels['images']
        img_annotations: List[Dict] = [img_annotation for img_annotation in self.labels['annotations'] if
                                       img_annotation['image_id'] == idx]
        img_path = os.path.join(self.img_dir, images[idx]['file_name'])
        image: torch.Tensor = read_image(str(img_path), ImageReadMode.RGB).float() / 255.0
        target: Dict = {}

        category_ids: List[int] = [img_annotation['category_id'] for img_annotation in img_annotations]
        bboxes: List[List[int]] = [ann['bbox'] for ann in img_annotations]

        image_id: int = images[idx]['id']  # must be unique for each image, used during evaluation
        for img_annotation in img_annotations:
            if img_annotation['category_id'] == 4:  # arrow
                keypoints: List[float] = img_annotation['keypoints']
                if image_id == 168:
                    self.plot_image_with_keypoints_and_bbox(image_tensor=image, keypoints=[[keypoints[3], keypoints[4]], [keypoints[0], keypoints[1]]], bbox=img_annotation['bbox'])

                bbox_head: List[int] = self.generate_synthetic_bbox(image=image, center_x=int(keypoints[3]),
                                                                    center_y=int(keypoints[4]), image_id=image_id)
                category_ids.append(10)
                bboxes.append(bbox_head)
                bbox_tail: List[int] = self.generate_synthetic_bbox(image=image, center_x=int(keypoints[0]),
                                                                    center_y=int(keypoints[1]), image_id=image_id)
                category_ids.append(11)
                bboxes.append(bbox_tail)

        boxes: torch.Tensor = torch.tensor(bboxes, dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
        boxes[:, 2:] += boxes[:, :2]

        labels: torch.Tensor = torch.tensor(category_ids, dtype=torch.int64)
        # label = 0 -> background
        area: torch.Tensor = torch.tensor([img_annotation['area'] for img_annotation in img_annotations])
        iscrowd: torch.Tensor = torch.tensor([img_annotation['iscrowd'] for img_annotation in img_annotations])
        target['boxes']: torch.Tensor = boxes
        target['labels']: torch.Tensor = labels
        target['image_id']: torch.Tensor = torch.tensor([image_id])
        target['area']: torch.Tensor = area
        target['iscrowd']: torch.Tensor = iscrowd
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return t.Image(image), target