import math

from core.image.bbox.bbox import ImageBoundingBox
from src.utils.bbox_vertices import bbox_vertices


def bbox_distance(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> float:
    bbox1_vertices, bbox2_vertices = bbox_vertices(bbox1=bbox1, bbox2=bbox2)

    distance: float = 0
    for bbox1_vertex in bbox1_vertices:
        for bbox2_vertex in bbox2_vertices:
            bbox_dist = math.sqrt((bbox1_vertex[0] - bbox2_vertex[0]) ** 2 + (bbox1_vertex[1] - bbox2_vertex[1]) ** 2)
            if bbox_dist < distance or distance == 0:
                distance = bbox_dist

    return distance
