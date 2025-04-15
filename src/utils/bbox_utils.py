import math
from typing import List, Tuple

from core.image.bbox.bbox import ImageBoundingBox


def bbox_distance(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> float:
    """
    Use Pythagoras' theorem to compute the distance between two bboxes
    """
    bbox1_vertices, bbox2_vertices = bbox_vertices(bbox1=bbox1, bbox2=bbox2)

    distance: float = 0
    for bbox1_vertex in bbox1_vertices:
        for bbox2_vertex in bbox2_vertices:
            bbox_dist = math.sqrt((bbox1_vertex[0] - bbox2_vertex[0]) ** 2 + (bbox1_vertex[1] - bbox2_vertex[1]) ** 2)
            if bbox_dist < distance or distance == 0:
                distance = bbox_dist

    return distance


def bbox_overlap(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> float:
    """
    Check how much of bbox2 is overlapped to bbox1
    """
    x_left: float = max(bbox1.top_left_x, bbox2.top_left_x)
    y_top: float = min(bbox1.top_left_y, bbox2.top_left_y)
    x_right: float = min(bbox1.bottom_right_x, bbox2.bottom_right_x)
    y_bottom: float = max(bbox1.bottom_right_y, bbox2.bottom_right_y)

    intersection_area: float = max(0, x_right - x_left) * max(0, y_top - y_bottom)
    area_element: float = (bbox1.bottom_right_x - bbox1.top_left_x) * (
            bbox1.top_left_y - bbox1.bottom_right_y)
    area_text: float = (bbox2.bottom_right_x - bbox2.top_left_x) * (
            bbox2.top_left_y - bbox2.bottom_right_y)

    overlap_text: float = intersection_area / area_text
    return overlap_text


def bbox_vertices(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Returns the vertices (as a list of coordinates (x, y) for each point) of two bboxes
    """
    bbox1_vertices: List[Tuple[float, float]] = [(bbox1.top_left_x, bbox1.top_left_y),
                                                 (bbox1.bottom_left_x, bbox1.bottom_left_y),
                                                 (bbox1.top_right_x, bbox1.top_right_y),
                                                 (bbox1.bottom_right_x, bbox1.bottom_right_y)]
    bbox2_vertices: List[Tuple[float, float]] = [(bbox2.top_left_x, bbox2.top_left_y),
                                                 (bbox2.bottom_left_x, bbox2.bottom_left_y),
                                                 (bbox2.top_right_x, bbox2.top_right_y),
                                                 (bbox2.bottom_right_x, bbox2.bottom_right_y)]

    return bbox1_vertices, bbox2_vertices


def bbox_split(bbox: ImageBoundingBox, direction: str, ratios: List[int]):
    pass
