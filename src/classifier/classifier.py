from core.classifier.classifier import Classifier
from core.image.image import Image
from src.classifier.preprocessing.processor import GNRMultiProcessor, Processor
from src.classifier.model import ClassifierCNN

import torch.nn as nn

class GNRClassifier(Classifier):
    
    def __init__(self, model_path: str, model: nn.Module = ClassifierCNN(), processor: Processor = GNRMultiProcessor()):
        super().__init__()
        if model_path is None:
            self.model = model
        else:
            self.model = model.load(model_path)
        self.model.eval()
        self.processor = processor
    
    def classify(self, image: Image) -> str:
        """
        Classify the image using the model and processor.
        
        :param image: The image to classify.
        
        :return: The classification result.
        """
        
        image = self.processor.process(image)
        
        raise NotImplemented()