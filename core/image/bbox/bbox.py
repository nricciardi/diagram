from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch import Tensor


@dataclass(frozen=True)
class ImageBoundingBox(ABC):
    category: str
    box: Tensor
    trust: float

    def __post_init__(self):
        if self.trust > 1 or self.trust < 0:
            raise ValueError("trust must be between 0 and 1")

    @property
    @abstractmethod
    def top_left_x(self) -> float:
        pass

    @property
    @abstractmethod
    def top_left_y(self) -> float:
        pass

    @property
    @abstractmethod
    def top_right_x(self) -> float:
        pass

    @property
    @abstractmethod
    def top_right_y(self) -> float:
        pass

    @property
    @abstractmethod
    def bottom_left_x(self) -> float:
        pass

    @property
    @abstractmethod
    def bottom_left_y(self) -> float:
        pass

    @property
    @abstractmethod
    def bottom_right_x(self) -> float:
        pass

    @property
    @abstractmethod
    def bottom_right_y(self) -> float:
        pass


