from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.image.image import Image
from core.utils.compatible_mixins import IdentifiableMixin, CompatibleDiagramsMixin
from core.utils.to_device import ToDeviceMixin


@dataclass
class Classifier(ToDeviceMixin, IdentifiableMixin, CompatibleDiagramsMixin, ABC):

    @abstractmethod
    def classify(self, image: Image) -> str:
        """
        Classify input image, in order to find the most probable diagram type

        :param image: input image
        :return: the most probable diagram type (diagram id)
        """

    @abstractmethod
    def compatible_diagrams(self) -> List[str]:
        raise NotImplemented()
