import logging
from dataclasses import dataclass
from typing import List, Tuple
from scipy.spatial import ConvexHull
from shapely.constructive import convex_hull
from shapely.geometry import Polygon
import torch
from matplotlib.image import BboxImage
from shapely import LineString, Polygon
import numpy as np
from core.image.bbox.bbox import ImageBoundingBox
from core.image.bbox.bbox2p import ImageBoundingBox2Points
from src.utils.bbox_utils import bbox_overlap, IoU
import cv2
from sklearn.cluster import DBSCAN
from typing import Optional


@dataclass(frozen=True)
class Arrow:

    x_head: int
    y_head: int
    x_tail: int
    y_tail: int

    bbox: ImageBoundingBox

    @classmethod
    def from_bboxes(cls, head_bbox: ImageBoundingBox, tail_bbox: ImageBoundingBox, arrow_bbox: ImageBoundingBox):
        # Compute the center of the head and tail bboxes
        x_head = int((head_bbox.top_left_x + head_bbox.bottom_right_x) // 2)
        y_head = int((head_bbox.top_left_y + head_bbox.bottom_right_y) // 2)
        x_tail = int((tail_bbox.top_left_x + tail_bbox.bottom_right_x) // 2)
        y_tail = int((tail_bbox.top_left_y + tail_bbox.bottom_right_y) // 2)

        return Arrow(x_head=x_head, y_head=y_head, x_tail=x_tail, y_tail=y_tail, bbox=arrow_bbox)

    def __greatest_convexhull(self, labels, points):
        unique_labels = set(labels) - {-1}

        max_area = 0
        largest_cluster_label = None
        largest_cluster_points = None
        largest_hull = None

        for label in unique_labels:
            cluster_points = points[labels == label]

            if len(cluster_points) >= 3:  # at least 3 points to build a Convex Hull
                try:
                    hull = ConvexHull(cluster_points)
                    area = hull.volume  # For 2D, 'volume' is area of Convex Hull
                    if area > max_area:
                        max_area = area
                        largest_cluster_label = label
                        largest_cluster_points = cluster_points
                        largest_hull = hull
                except:
                    continue  # ignore invalid cluster for convex hull

        largest_cluster_label_with_largest_convexhull = largest_cluster_label.copy()
        largest_cluster_points_with_largest_convexhull = largest_cluster_points.copy()

        logging.debug(f"label of largest cluster: {largest_cluster_label_with_largest_convexhull}")
        logging.debug(f"n. points: {largest_cluster_points_with_largest_convexhull}")

        return largest_hull

    def compute_opposite_point(self) -> tuple[bool, Optional[float], Optional[float]]:
        """
            Determines if an arrow has head and tail near the same sides.
            Returns:
                :bool: True, if head and tail on the same side
                :Optional[float]: x
                :Optional[float]: y
        """
        lower_side_y = self.bbox.bottom_left_y
        upper_side_y = self.bbox.top_left_y
        left_side_x = self.bbox.top_left_x
        right_side_x = self.bbox.top_right_x

        mean_points = [
            (int((self.bbox.bottom_left_x + self.bbox.bottom_right_x) / 2), self.bbox.bottom_left_y),
            (self.bbox.top_right_x, int((self.bbox.top_right_y + self.bbox.bottom_right_y) / 2)),
            (int((self.bbox.top_left_x + self.bbox.top_right_x) / 2), self.bbox.top_left_y),
            (self.bbox.bottom_left_x, int((self.bbox.top_left_y + self.bbox.bottom_left_y) / 2),)
        ]

        vertices = [
            (self.bbox.top_left_x, self.bbox.top_left_y),
            (self.bbox.bottom_left_x, self.bbox.bottom_left_y),
            (self.bbox.bottom_right_x, self.bbox.bottom_right_y),
            (self.bbox.top_right_x, self.bbox.top_right_y),
        ]

        head_distances = [
            abs(self.y_head - lower_side_y),
            abs(self.x_head - right_side_x),
            abs(self.y_head - upper_side_y),
            abs(self.x_head - left_side_x)
        ]

        tail_distances = [
            abs(self.y_tail - lower_side_y),
            abs(self.x_tail - right_side_x),
            abs(self.y_tail - upper_side_y),
            abs(self.x_tail - left_side_x)
        ]

        convert_table = {
            2: 0,
            5: 1,
            8: 2,
            6: 3
        }

        head_min_distance = min(head_distances)
        head_min_index = head_distances.index(head_min_distance)

        tail_min_distance = min(tail_distances)
        tail_min_index = tail_distances.index(tail_min_distance)

        if head_min_index == tail_min_index:
            side_index = (head_min_index + 2) % 4
            return True, mean_points[side_index][0], mean_points[side_index][1]

        if (abs(head_min_index - tail_min_index)) == 1 or (abs(head_min_index - tail_min_index)) == 3:
            minV = min(head_min_index, tail_min_index)
            maxV = max(head_min_index, tail_min_index) * 2
            index = convert_table[minV + maxV]
            return True, vertices[index][0], vertices[index][1]

        return False, None, None


    def is_self(self) -> bool:
        s, _, _ = self.compute_opposite_point()

        return s

    def distance_to_bbox(self, other: ImageBoundingBox) -> float:
        is_self, x, y = self.compute_opposite_point()

        if is_self:
            assert x is not None
            assert y is not None
            arrow_line = LineString(coordinates=[[self.x_tail, self.y_tail], [x, y], [self.x_head, self.y_head]])
        else:
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

def compute_arrows(arrow_bboxes: List[ImageBoundingBox], head_bboxes: List[ImageBoundingBox], tail_bboxes: List[ImageBoundingBox]) \
        -> Tuple[List[Arrow], List[ImageBoundingBox], List[ImageBoundingBox], List[ImageBoundingBox]]:
    """
    arrow_bboxes: bboxes of arrows
    arrow_bboxes: bboxes of arrow heads
    arrow_bboxes: bboxes of arrow tails

    :returns: List of Arrow objects, remaining arrow bboxes, remaining head bboxes, remaining tail bboxes
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

        arrows.append(Arrow.from_bboxes(head_bbox=head_bbox, tail_bbox=tail_bbox, arrow_bbox=arrow))

    arrow_bboxes_to_recover = []
    for arrow_bbox in arrow_bboxes:
        insert: bool = True
        for arrow in arrows:
            if arrow.bbox.eq(arrow_bbox):
                insert = False
                break

        if insert:
            arrow_bboxes_to_recover.append(arrow_bbox)

    assert len(arrow_bboxes_to_recover) == len(arrow_bboxes) - len(arrows)

    return arrows, arrow_bboxes_to_recover, head_bboxes, tail_bboxes