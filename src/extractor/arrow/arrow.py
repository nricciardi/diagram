from dataclasses import dataclass
from typing import List

import torch
from matplotlib.image import BboxImage

from core.image.bbox.bbox import ImageBoundingBox
from core.image.bbox.bbox2p import ImageBoundingBox2Points
from src.utils.bbox_utils import bbox_overlap, IoU


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

    arrows: List[Arrow] = []

    # Sort bboxes based on the number of overlaps with arrow bboxes
    head_bboxes = sorted(
        head_bboxes,
        key=lambda bbox: sum([
            1 if bbox_overlap(bbox, arrow_bbox) > 0 else 0 for arrow_bbox in arrow_bboxes
        ])
    )
    tail_bboxes = sorted(
        tail_bboxes,
        key=lambda bbox: sum([
            1 if bbox_overlap(bbox, arrow_bbox) > 0 else 0 for arrow_bbox in arrow_bboxes
        ])
    )
    arrow_bboxes = sorted(
        arrow_bboxes,
        key=lambda bbox: len([
            1 if bbox_overlap(bbox, head_bbox) > 0 else 0 for head_bbox in head_bboxes
        ] + [
            1 if bbox_overlap(bbox, tail_bbox) > 0 else 0 for tail_bbox in tail_bboxes
        ])
    )
    
    def get_most_certain_bbox(bboxes_part: List[ImageBoundingBox], arrow_bbox: ImageBoundingBox) -> ImageBoundingBox:
        """
        Get the most certain bbox from bboxes_part that overlaps with arrow_bbox.
        """
        max_overlap = 0
        most_certain_bbox = None
        
        for bbox in bboxes_part:
            overlap = bbox_overlap(bbox, arrow_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                most_certain_bbox = bbox
        
        return most_certain_bbox

    for arrow in arrow_bboxes:
        head_bbox = get_most_certain_bbox(head_bboxes, arrow)
        tail_bbox = get_most_certain_bbox(tail_bboxes, arrow)

        if head_bbox is None or tail_bbox is None:
            continue

        # Compute the center of the head and tail bboxes
        x_head = (head_bbox.top_left_x + head_bbox.bottom_right_x) // 2
        y_head = (head_bbox.top_left_y + head_bbox.bottom_right_y) // 2
        x_tail = (tail_bbox.top_left_x + tail_bbox.bottom_right_x) // 2
        y_tail = (tail_bbox.top_left_y + tail_bbox.bottom_right_y) // 2

        arrows.append(Arrow(x_head, y_head, x_tail, y_tail, arrow))

    return arrows