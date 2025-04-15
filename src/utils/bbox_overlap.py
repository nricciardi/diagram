from core.image.bbox.bbox import ImageBoundingBox


def bbox_overlap(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> float:
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
