from core.image.bbox.bbox import ImageBoundingBox
from typing import List, Tuple
import math


def bbox_distance(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> float:
    bbox1_vertices: List[Tuple[float, float]] = [(bbox1.top_left_x, bbox1.top_left_y),
                                                 (bbox1.bottom_left_x, bbox1.bottom_left_y),
                                                 (bbox1.top_right_x, bbox1.top_right_y),
                                                 (bbox1.bottom_right_x, bbox1.bottom_right_y)]
    bbox2_vertices: List[Tuple[float, float]] = [(bbox2.top_left_x, bbox2.top_left_y),
                                                 (bbox2.bottom_left_x, bbox2.bottom_left_y),
                                                 (bbox2.top_right_x, bbox2.top_right_y),
                                                 (bbox2.bottom_right_x, bbox2.bottom_right_y)]

    distance: float = 0
    for bbox1_vertex in bbox1_vertices:
        for bbox2_vertex in bbox2_vertices:
            bbox_dist = math.sqrt((bbox1_vertex[0] - bbox2_vertex[0]) ** 2 + (bbox1_vertex[1] - bbox2_vertex[1]) ** 2)
            if bbox_dist < distance or distance == 0:
                distance = bbox_dist

    return distance