from abc import ABC, abstractmethod

from src.representation.representation import DiagramRepresentation
from src.image import Image


class Extractor(ABC):

    def __init__(self, identifier: str):
        self._identifier = identifier

    @abstractmethod
    def extract(self, image: Image, diagram_id: str) -> DiagramRepresentation:
        """
        Extract agnostic representation of a diagram (image)

        :param diagram_id: diagram identifier
        :param image: input image
        :return: agnostic representation
        """

