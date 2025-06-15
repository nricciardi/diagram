from dataclasses import dataclass

from matplotlib.image import BboxImage

from core.image.bbox.bbox2p import ImageBoundingBox2Points


@dataclass(frozen=True)
class Arrow:

    x_head: int
    y_head: int
    x_tail: int
    y_tail: int



