from typing import Tuple, List

from core.image.bbox.bbox import ImageBoundingBox


def bbox_vertices(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]]]:
    bbox1_vertices: List[Tuple[float, float]] = [(bbox1.top_left_x, bbox1.top_left_y),
                                                 (bbox1.bottom_left_x, bbox1.bottom_left_y),
                                                 (bbox1.top_right_x, bbox1.top_right_y),
                                                 (bbox1.bottom_right_x, bbox1.bottom_right_y)]
    bbox2_vertices: List[Tuple[float, float]] = [(bbox2.top_left_x, bbox2.top_left_y),
                                                 (bbox2.bottom_left_x, bbox2.bottom_left_y),
                                                 (bbox2.top_right_x, bbox2.top_right_y),
                                                 (bbox2.bottom_right_x, bbox2.bottom_right_y)]

    return bbox1_vertices, bbox2_vertices
