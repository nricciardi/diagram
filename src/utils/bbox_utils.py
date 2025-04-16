import math
from typing import List, Tuple
import numpy as np

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
                                                 (bbox1.top_right_x, bbox1.top_right_y),
                                                 (bbox1.bottom_right_x, bbox1.bottom_right_y),
                                                 (bbox1.bottom_left_x, bbox1.bottom_left_y)]
    bbox2_vertices: List[Tuple[float, float]] = [(bbox2.top_left_x, bbox2.top_left_y),
                                                 (bbox2.top_right_x, bbox2.top_right_y),
                                                 (bbox2.bottom_right_x, bbox2.bottom_right_y),
                                                 (bbox2.bottom_left_x, bbox2.bottom_left_y)]

    return bbox1_vertices, bbox2_vertices


def interpolate(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> np.ndarray:
    return (1 - t) * np.array(p1) + t * np.array(p2)


def bbox_split(bbox: ImageBoundingBox, direction: str, ratios: List[float], arrow_head: str) -> List[List[Tuple]]:
    """
    Splits the arrow bbox into three parts along the given direction and according to the given ratios
    """

    vertices, other_vertices = bbox_vertices(bbox1=bbox, bbox2=bbox)

    if direction == 'height':
        if arrow_head == 'down':
            left = [vertices[0], vertices[3]]
            right = [vertices[1], vertices[2]]
        else:
            left = [vertices[3], vertices[0]]
            right = [vertices[2], vertices[1]]

        boxes = []
        t_start = 0

        for ratio in ratios:
            t_end = t_start + ratio

            # Interpolate along the left and right edges
            left_top = interpolate(left[0], left[1], t_start).tolist()
            left_bottom = interpolate(left[0], left[1], t_end).tolist()
            right_top = interpolate(right[0], right[1], t_start).tolist()
            right_bottom = interpolate(right[0], right[1], t_end).tolist()

            # Create sub-box (clockwise)
            sub_box = [
                tuple(left_top),
                tuple(right_top),
                tuple(right_bottom),
                tuple(left_bottom)
            ]
            boxes.append(sub_box)

            t_start = t_end
    else:
        if arrow_head == 'right':
            top_edge = [vertices[0], vertices[1]]
            bottom_edge = [vertices[3], vertices[2]]
        else:
            top_edge = [vertices[1], vertices[0]]
            bottom_edge = [vertices[2], vertices[3]]

        boxes = []
        t_start = 0

        for ratio in ratios:
            t_end = t_start + ratio

            # Interpolate on top and bottom edges
            top_left = interpolate(top_edge[0], top_edge[1], t_start)
            top_right = interpolate(top_edge[0], top_edge[1], t_end)
            bottom_left = interpolate(bottom_edge[0], bottom_edge[1], t_start)
            bottom_right = interpolate(bottom_edge[0], bottom_edge[1], t_end)

            sub_box = [
                tuple(top_left),
                tuple(top_right),
                tuple(bottom_right),
                tuple(bottom_left)
            ]
            boxes.append(sub_box)

            t_start = t_end

    return boxes
