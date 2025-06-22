from dataclasses import dataclass
from typing import List

import torch
from matplotlib.image import BboxImage

from core.image.bbox.bbox import ImageBoundingBox
from core.image.bbox.bbox2p import ImageBoundingBox2Points


@dataclass(frozen=True)
class Arrow:

    x_head: int
    y_head: int
    x_tail: int
    y_tail: int

    bbox: ImageBoundingBox


def compute_arrows(arrow_bboxes: List[ImageBoundingBox], head_bboxes: List[ImageBoundingBox], tail_bboxes: List[ImageBoundingBox]) -> List[Arrow]:
    """
    arrow_bboxes: bboxes of arrows
    arrow_bboxes: bboxes of arrow heads
    arrow_bboxes: bboxes of arrow tails

    Returns: |arrow_bboxes| arrows
    """

    # TODO






