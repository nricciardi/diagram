from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch import Tensor


@dataclass(frozen=True)
class ImageBoundingBox(ABC):
    """
    Abstract base class representing an image bounding box.
    Attributes:
        category (int): The category or label associated with the bounding box.
        box (Tensor): The tensor representation of the bounding box coordinates.
        trust (float): A confidence score for the bounding box, must be between 0 and 1.
    Methods:
        top_left_x() -> float:
            Abstract property to get the x-coordinate of the top-left corner of the bounding box.
        top_left_y() -> float:
            Abstract property to get the y-coordinate of the top-left corner of the bounding box.
        top_right_x() -> float:
            Abstract property to get the x-coordinate of the top-right corner of the bounding box.
        top_right_y() -> float:
            Abstract property to get the y-coordinate of the top-right corner of the bounding box.
        bottom_left_x() -> float:
            Abstract property to get the x-coordinate of the bottom-left corner of the bounding box.
        bottom_left_y() -> float:
            Abstract property to get the y-coordinate of the bottom-left corner of the bounding box.
        bottom_right_x() -> float:
            Abstract property to get the x-coordinate of the bottom-right corner of the bounding box.
        bottom_right_y() -> float:
            Abstract property to get the y-coordinate of the bottom-right corner of the bounding box.
    Raises:
        ValueError: If the `trust` attribute is not between 0 and 1.
    """

    category: int
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


