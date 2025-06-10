from torch.utils.data import Dataset
from core.image.tensor_image import TensorImage, Image
import json
from src.classifier.preprocessing.processor import Processor, GNRMultiProcessor

class DatasetClassifier(Dataset):
    """
    A custom dataset class for image classification tasks.
    
    :param images: A list of image tensors.
    :param labels: A list of labels corresponding to the images.
    """
    
    def __init__(self, preprocessor: Processor = GNRMultiProcessor(), json_path: str = "dataset/classifier/labels.json", base_image_path: str = "dataset/classifier/all/"):
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
        self.images = []
        self.labels = []
        idx = 0
        for key, value in data.items():
            self.images.append(key)
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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        :param idx: The index of the item to retrieve.
        
        :return: A tuple containing the image tensor and its corresponding label.
        """
        image: Image = TensorImage.from_str(self.base_image_path + self.images[idx])
        image = self.preprocessor.process(image)
        return image.as_tensor(), self.labels[idx]