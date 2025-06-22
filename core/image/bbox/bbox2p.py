from core.image.bbox.bbox import ImageBoundingBox
from dataclasses import dataclass
import torch
from typing import Self


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
