from abc import ABC, abstractmethod
from torch import Tensor
from typing import Self
from core.utils.to_device import ToDeviceMixin


class Image(ToDeviceMixin, ABC):
    """
    Wrap class for images
    """

    @classmethod
    @abstractmethod
    def from_str(cls, path: str) -> Self:
        pass

    @abstractmethod
    def as_tensor(self) -> Tensor:
        pass