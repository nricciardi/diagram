from core.image.bbox.bbox import ImageBoundingBox
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageBoundingBox2Points(ImageBoundingBox):
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

