from abc import ABC, abstractmethod
from typing import List

from core.representation.representation import DiagramRepresentation
from core.image.image import Image
from core.utils.compatible_mixins import CompatibleDiagramsMixin, IdentifiableMixin


class Extractor(IdentifiableMixin, CompatibleDiagramsMixin, ABC):

    @abstractmethod
    async def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:
        """
        Extract agnostic representation of a diagram (image)

        :param diagram_id: diagram identifier
        :param image: input image
        :return: agnostic representation
        """

