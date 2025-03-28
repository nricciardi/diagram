from abc import ABC, abstractmethod
from typing import List

from core.representation.representation import DiagramRepresentation
from core.image import Image


class Extractor(ABC):

    def __init__(self, identifier: str):
        self._identifier = identifier

    @abstractmethod
    def compatible_diagrams(self) -> List[str]:
        pass

    @abstractmethod
    def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:
        """
        Extract agnostic representation of a diagram (image)

        :param diagram_id: diagram identifier
        :param image: input image
        :return: agnostic representation
        """

