import json
import os
from typing import Tuple, Dict, List

from torch.utils.data import Dataset
from torchvision import tv_tensors as t
from torchvision.io import read_image, ImageReadMode
import torch


class ObjectDetectionDataset(Dataset):
    """
    Dataset class for fine-tuning the object detection network

    :param annotations_file: path to the labels file
    :param img_dir: path to the images directory
    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        with open(annotations_file, 'r') as f:
            self.labels = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels['images'])

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
        image = read_image(str(img_path), ImageReadMode.RGB).float() / 255.0
        target: Dict = {}

        boxes = torch.tensor([ann['bbox'] for ann in img_annotations], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
        boxes[:, 2:] += boxes[:, :2]

        labels: torch.Tensor = torch.tensor([img_annotation['category_id'] for img_annotation in img_annotations], dtype=torch.int64)
        # label = 0 -> background
        image_id: int = images[idx]['id']  # must be unique for each image, used during evaluation
        area: torch.Tensor = torch.tensor([img_annotation['area'] for img_annotation in img_annotations])
        iscrowd: torch.Tensor = torch.tensor([img_annotation['iscrowd'] for img_annotation in img_annotations])
        target['boxes'] = boxes
        target['labels'] = labels
        # target['image_id'] = image_id
        target['image_id'] = torch.tensor([image_id])
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target
