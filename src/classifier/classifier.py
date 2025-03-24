from abc import ABC, abstractmethod

from src.diagram import Diagram
from src.image import Image

class Classifier(ABC):

    @abstractmethod
    def classify(self, image: Image) -> str:
        """
        Classify input image, in order to find the most probable diagram type

        :param image: input image
        :return: the most probable diagram type (diagram id)
        """

