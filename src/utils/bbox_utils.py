import math
from typing import List, Tuple
import numpy as np

from core.image.bbox.bbox import ImageBoundingBox


def bbox_distance(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> float:
    """
    Calculate the shortest distance between two bounding boxes using Pythagoras' theorem.
    This function computes the Euclidean distance between all pairs of vertices from 
    two bounding boxes and returns the smallest distance.
    Args:
        bbox1 (ImageBoundingBox): The first bounding box.
        bbox2 (ImageBoundingBox): The second bounding box.
    Returns:
        float: The shortest Euclidean distance between the two bounding boxes.
    """
    bbox1_vertices, bbox2_vertices = bbox_vertices(bbox1=bbox1, bbox2=bbox2)

    distance: float = 0
    for bbox1_vertex in bbox1_vertices:
        for bbox2_vertex in bbox2_vertices:
            bbox_dist = math.sqrt((bbox1_vertex[0] - bbox2_vertex[0]) ** 2 + (bbox1_vertex[1] - bbox2_vertex[1]) ** 2)
            if bbox_dist < distance or distance == 0:
                distance = bbox_dist

    return distance


def bbox_overlap(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox, two_wrt_one: bool = True) -> float:
    """
    Calculates the overlap percentage between two bounding boxes.
    This function computes the percentage of overlap between two bounding boxes
    (`bbox1` and `bbox2`). The overlap is calculated as the area of intersection
    divided by the area of one of the bounding boxes, depending on the value of
    `two_wrt_one`.
    Args:
        bbox1 (ImageBoundingBox): The first bounding box.
        bbox2 (ImageBoundingBox): The second bounding box.
        two_wrt_one (bool, optional): If True, the overlap is calculated as the
            intersection area divided by the area of `bbox2`. If False, the overlap
            is calculated as the intersection area divided by the area of `bbox1`.
            Defaults to True.
    Returns:
        float: The percentage of overlap between the two bounding boxes.
    Raises:
        ValueError: If `bbox2` is invalid (has zero or negative area) and
            `two_wrt_one` is True.
        ValueError: If `bbox1` is invalid (has zero or negative area) and
            `two_wrt_one` is False.
    """
    x_left: float = max(bbox1.top_left_x, bbox2.top_left_x)
    y_top: float = min(bbox1.top_left_y, bbox2.top_left_y)
    x_right: float = min(bbox1.bottom_right_x, bbox2.bottom_right_x)
    y_bottom: float = max(bbox1.bottom_right_y, bbox2.bottom_right_y)

    intersection_area: float = max(0., x_right - x_left) * max(0., y_top - y_bottom)
    area_bbox1: float = (bbox1.bottom_right_x - bbox1.top_left_x) * (
            bbox1.top_left_y - bbox1.bottom_right_y)
    area_bbox2: float = (bbox2.bottom_right_x - bbox2.top_left_x) * (
            bbox2.top_left_y - bbox2.bottom_right_y)

    if area_bbox2 <= 0 and two_wrt_one:
        raise ValueError("bbox2 is an invalid bbox")
    elif area_bbox1 <= 0 and not two_wrt_one:
        raise ValueError("bbox1 is an invalid bbox")

    overlap_percentage: float = intersection_area / (area_bbox2 if two_wrt_one else area_bbox1)
    return overlap_percentage


def bbox_vertices(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Computes the vertices of two bounding boxes and returns them as lists of coordinate tuples.
    Args:
        bbox1 (ImageBoundingBox): The first bounding box, containing attributes for its corner coordinates.
        bbox2 (ImageBoundingBox): The second bounding box, containing attributes for its corner coordinates.
    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]: 
            A tuple containing two lists:
            - The first list contains the vertices of `bbox1` as (x, y) coordinate tuples.
            - The second list contains the vertices of `bbox2` as (x, y) coordinate tuples.
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
    """
    Linearly interpolates between two points in 2D space.
    Parameters:
        p1 (Tuple[float, float]): The first point as a tuple of (x, y) coordinates.
        p2 (Tuple[float, float]): The second point as a tuple of (x, y) coordinates.
        t (float): The interpolation factor, where 0 <= t <= 1. 
                   A value of 0 returns p1, a value of 1 returns p2, 
                   and values in between return a point along the line segment.
    Returns:
        np.ndarray: The interpolated point as a NumPy array of shape (2,).
    """

    return (1 - t) * np.array(p1) + t * np.array(p2)


def bbox_split(bbox: ImageBoundingBox, ratios: List[float], arrow_head: str) -> List[List[Tuple]]:
    """
    Splits a bounding box into smaller sub-boxes based on the specified direction, ratios, and arrow head orientation.
    Args:
        bbox (ImageBoundingBox): The bounding box to be split, defined by its vertices.
        ratios (List[float]): A list of ratios that determine the size of each sub-box along the split direction.
                              The sum of the ratios should equal 1.
        arrow_head (str): The orientation of the arrow head, which determines the order of the split.
                          For vertical splits, it can be 'down' or 'up'.
                          For horizontal splits, it can be 'right' or 'left'.
    Returns:
        List[List[Tuple]]: A list of sub-boxes, where each sub-box is represented as a list of tuples.
                           Each tuple corresponds to a vertex of the sub-box in clockwise order.
    Raises:
        ValueError: If the direction is not 'vertically' or 'horizontally'.
                    If the arrow_head is not valid for the given direction.
                    If the sum of ratios does not equal 1.
    Notes:
        - While it is technically possible to pass 2 points bboxes as parameters, to have more accurate results,
        it is suggested to use 4 points bboxes
        - Arrow_head is needed both to understand which sides to cut and to distinguish which split is the source
        and which is the target
        - In case of multiple heads (or no heads at all), a random one (among the two possible for the type of
        split) should be passed
    """

    if arrow_head not in ['down', 'up', 'left', 'right']:
        raise ValueError(f'Arrow_head should be down, up, left or right; got {arrow_head} instead')

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f'Sum of ratios should equal 1, got {sum(ratios)} instead')

    # bbox 4 points (not aligned)
    # posizione testa freccia (4 direzioni)
    # 2 teste -> me ne dai una a caso

    # testa freccia serve per distinguere source e target (4 direzioni)
    # direzione serve per capire lungo quali lati tagliere

    # Riccio's improvement
    # testa freccia fa entrambe (4 direzioni)

    # Adesso
    # testa freccia: 0, 1, 2 -> 1 OK, 0/2 -> me ne dai una a caso
    # hp: direzione esiste sempre ed è unica (aka freccia è una retta con 0, 1, 2 teste)

    # Cosa abbiamo
    # bbox 2 punti senza conoscenza del tipo di freccia
    # cosa ci serve: bbox 4 punti + come è orientata la freccia + dove sono le teste
    # come ottenere la testa:

    # hp: rete (fatta da noi o **già fatta**) che tira fuori test* e cod* della freccia da una bbox 2 punti
    # freccia -> retta che passa per test* o cod* (easy)
    # bbox 4 punti -> la costruiamo noi a partire da testa e coda con offset decisi da noi (iperparametri)

    vertices, other_vertices = bbox_vertices(bbox1=bbox, bbox2=bbox)

    if arrow_head in ['down', 'up']:
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


def bbox_relative_position(first_bbox: ImageBoundingBox, second_bbox: ImageBoundingBox) -> str:
    """
    Determines the relative position of a second bounding box with respect to a first bounding box.
    Args:
        first_bbox (ImageBoundingBox): The reference bounding box.
        second_bbox (ImageBoundingBox): The bounding box whose position is to be determined.
    Returns:
        str: A string indicating the relative position of the second bounding box.
             Possible values are:
             - "right": The second bounding box is to the right of the first bounding box.
             - "left": The second bounding box is to the left of the first bounding box.
             - "down": The second bounding box is below the first bounding box.
             - "up": The second bounding box is above the first bounding box.
    """

    if second_bbox.top_left_x > first_bbox.top_left_x:
        return "right"
    elif second_bbox.top_left_x < first_bbox.top_left_x:
        return "left"
    else:
        if second_bbox.bottom_right_y < second_bbox.top_right_y:
            return "down"
        else:
            return "up"
