from dataclasses import dataclass
from typing import Set, List

import torch
from matplotlib.image import BboxImage

from core.image.bbox.bbox2p import ImageBoundingBox2Points


@dataclass(frozen=True)
class Arrow:

    x_head: int
    y_head: int
    x_tail: int
    y_tail: int

    bbox: ImageBoundingBox2Points


def compute_arrows(arrow_bboxes: List[ImageBoundingBox2Points], head_bboxes: List[ImageBoundingBox2Points], tail_bboxes: List[ImageBoundingBox2Points]) -> Set[Arrow]:
    """
    arrow_bboxes: bboxes of arrows
    arrow_bboxes: bboxes of arrow heads
    arrow_bboxes: bboxes of arrow tails

    Returns: |arrow_bboxes| arrows
    """

    # TODO






