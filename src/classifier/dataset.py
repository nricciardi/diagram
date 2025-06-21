from torch.utils.data import Dataset
from core.image.tensor_image import TensorImage, Image
import json, random
from src.classifier.preprocessing.processor import Processor, GNRMultiProcessor, ReverseProcessor

from src.wellknown_diagram import WellKnownDiagram
from dataclasses import dataclass, field
from typing import List

class ImagePath:

        def __init__(self, name: str, reversed: bool = False):
            """
            Initialize the ImagePath with a path and an optional reversed flag.
            
            :param name: The name of the image.
            :param path: The path to the image.
            :param reversed: Whether the image should be reversed (default is False).
            """
            self.name = name
            self.reversed = reversed
        
        def get(self, base_path: str) -> Image:
            """
            Get the image from the path.
            
            :return: An Image object loaded from the path.
            """
            image: Image = TensorImage.from_str(base_path + self.name)
            if self.reversed:
                image = ReverseProcessor().process(image)
            return image

class DatasetClassifier(Dataset):
    """
    A custom dataset class for image classification tasks.
    
    :param images: A list of image tensors.
    :param labels: A list of labels corresponding to the images.
    """


    def __init__(self, preprocessor: Processor = GNRMultiProcessor(), json_path: str = "dataset/classifier/labels.json", base_image_path: str = "dataset/classifier/all/", undersample_classes: list[str] = [], upsample_classes: list[str] = []):
        """
        Initialize the dataset with images and labels.
        
        :param json_path: Path to the JSON file containing image paths and labels.
        """
        self.base_image_path = base_image_path
        self.preprocessor = preprocessor
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        freqs = {}
        self.classes = {}
        self.images: list[ImagePath] = []
        self.labels = []
        idx = 0
        for key, value in data.items():
            if value in undersample_classes and random.random() < 0.3:
                continue
            if value in upsample_classes and random.random() < 0.3:
                self.images.append(ImagePath(key, reversed=True))
                self.labels.append(value)
            self.images.append(ImagePath(key))
            self.labels.append(value)
            if value not in self.classes:
                self.classes[value] = idx
                idx += 1
            freqs[self.classes[value]] = freqs.get(self.classes[value], 0) + 1
        weights = [freqs[classId] for classId in self.classes.values()]

        # Calculate the inverse of the weights
        total = sum(weights)
        freqs = [w / total for w in weights]
        inv_freqs = [1 / w for w in freqs]
        sum_inv = sum(inv_freqs)
        self.weights = [w / sum_inv for w in inv_freqs]
        self.classes = list(set([
            WellKnownDiagram.from_string(label) for label in self.labels
        ]))
        self.classes.sort(key=lambda c: c.value)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        :param idx: The index of the item to retrieve.
        
        :return: A tuple containing the image tensor and its corresponding label.
        """
        image: Image = self.images[idx].get(self.base_image_path)
        image = self.preprocessor.process(image)
        return image.as_tensor(), self.labels[idx]