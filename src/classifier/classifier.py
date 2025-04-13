from core.classifier.classifier import Classifier
from core.image.image import Image


class GNRClassifier(Classifier):
    def classify(self, image: Image) -> str:
        raise NotImplemented()