from abc import ABC, abstractmethod
from torch import Tensor

from core.utils.to_device import ToDeviceMixin


class Image(ToDeviceMixin, ABC):
    """
    Wrap class for images
    """

    @staticmethod
    @abstractmethod
    def from_str(path: str) -> 'Image':
        pass

    @abstractmethod
    def as_tensor(self) -> Tensor:
        pass