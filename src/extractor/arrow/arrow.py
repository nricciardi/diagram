from dataclasses import dataclass
from typing import List

import torch
from matplotlib.image import BboxImage
from shapely import LineString, Polygon

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

    @classmethod
    def from_bboxes(cls, head_bbox: ImageBoundingBox, tail_bbox: ImageBoundingBox, arrow: ImageBoundingBox):
        # Compute the center of the head and tail bboxes
        x_head = int((head_bbox.top_left_x + head_bbox.bottom_right_x) // 2)
        y_head = int((head_bbox.top_left_y + head_bbox.bottom_right_y) // 2)
        x_tail = int((tail_bbox.top_left_x + tail_bbox.bottom_right_x) // 2)
        y_tail = int((tail_bbox.top_left_y + tail_bbox.bottom_right_y) // 2)
        return Arrow(x_head=x_head, y_head=y_head, x_tail=x_tail, y_tail=y_tail, bbox=arrow)

    def is_self(self) -> bool:
        pass # TODO backlog

    def distance_to_bbox(self, other: ImageBoundingBox) -> float:
        arrow_line = LineString(coordinates=[[self.x_tail, self.y_tail], [self.x_head, self.y_head]])
        other_poly = Polygon([[other.top_left_x, other.top_left_y], [other.top_right_x, other.top_right_y],
                              [other.bottom_right_x, other.bottom_right_y], [other.bottom_left_x, other.bottom_left_y]])
        return arrow_line.distance(other_poly)


def get_most_certain_bbox(bboxes_part: List[ImageBoundingBox], arrow_bbox: ImageBoundingBox) -> tuple[ImageBoundingBox, int]:
    """
    Get the most certain bbox from bboxes_part that overlaps with arrow_bbox.
    """
    max_overlap = 0
    most_certain_bbox = (None, None)

    for i, bbox in enumerate(bboxes_part):
        overlap = bbox_overlap(bbox, arrow_bbox)
        if overlap > max_overlap:
            max_overlap = overlap
            most_certain_bbox = (bbox, i)

    return most_certain_bbox

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

    for arrow in arrow_bboxes:
        head_bbox, i = get_most_certain_bbox(head_bboxes, arrow)
        tail_bbox, j = get_most_certain_bbox(tail_bboxes, arrow)
        if head_bbox is None or tail_bbox is None:
            continue
        
        head_bboxes.pop(i)
        tail_bboxes.pop(j)

        arrows.append(Arrow.from_bboxes(head_bbox=head_bbox, tail_bbox=tail_bbox, arrow=arrow))

    return arrows