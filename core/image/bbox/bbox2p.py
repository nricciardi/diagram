from core.image.bbox.bbox import ImageBoundingBox
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ImageBoundingBox2Points(ImageBoundingBox):
    """
    A class representing a bounding box for an image, defined by two points:
    the top-left and bottom-right corners.
    This class provides properties to access the coordinates of the four corners
    of the bounding box (top-left, top-right, bottom-left, bottom-right) in terms
    of their x and y values.
    """

    @property
    def top_left_x(self) -> float:
        return float(self.box[0])

    @property
    def top_left_y(self) -> float:
        return float(self.box[1])

    @property
    def top_right_x(self) -> float:
        return float(self.box[2])

    @property
    def top_right_y(self) -> float:
        return float(self.box[1])

    @property
    def bottom_left_x(self) -> float:
        return float(self.box[0])

    @property
    def bottom_left_y(self) -> float:
        return float(self.box[3])

    @property
    def bottom_right_x(self) -> float:
        return float(self.box[2])

    @property
    def bottom_right_y(self) -> float:
        return float(self.box[3])

    def area(self) -> float:
        assert self.top_right_x >= self.top_left_x
        # assert (self.top_right_x - self.top_left_x) == (self.bottom_right_x - self.bottom_left_x)

        assert self.top_right_y <= self.bottom_right_y
        # assert (self.top_right_y - self.bottom_right_y) == (self.top_left_y - self.bottom_left_y)

        return (self.top_right_x - self.top_left_x) * (self.bottom_right_y - self.top_right_y)

    def distance(self, other: ImageBoundingBox) -> float:

        # Distance along x
        if self.bottom_right_x < other.top_left_x:
            dx = other.top_left_x - self.bottom_right_x
        elif other.bottom_right_x < self.top_left_x:
            dx = self.top_left_x - other.bottom_right_x
        else:
            dx = 0  # overlap along x

        # Distance along y
        if self.bottom_right_y < other.top_left_y:
            dy = other.top_left_y - self.bottom_right_y
        elif other.bottom_right_y < self.top_left_y:
            dy = self.top_left_y - other.bottom_right_y
        else:
            dy = 0  # overlap along y

        return (dx ** 2 + dy ** 2) ** 0.5