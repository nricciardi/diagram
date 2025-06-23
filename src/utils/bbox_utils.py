import math
from typing import List, Tuple
import numpy as np

from core.image.bbox.bbox import ImageBoundingBox

from shapely.geometry import LineString
from shapely.ops import split
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    y_top: float = max(bbox1.top_left_y, bbox2.top_left_y)
    x_right: float = min(bbox1.bottom_right_x, bbox2.bottom_right_x)
    y_bottom: float = min(bbox1.bottom_right_y, bbox2.bottom_right_y)

    intersection_area: float = max(0., x_right - x_left) * max(0., y_bottom - y_top)
    area_bbox1: float = bbox1.area()
    area_bbox2: float = bbox2.area()

    if area_bbox2 <= 0 and two_wrt_one:
        raise ValueError("bbox2 is an invalid bbox")
    elif area_bbox1 <= 0 and not two_wrt_one:
        raise ValueError("bbox1 is an invalid bbox")

    overlap_percentage: float = intersection_area / (area_bbox2 if two_wrt_one else area_bbox1)
    return overlap_percentage


def bbox_vertices(bbox1: ImageBoundingBox, bbox2: ImageBoundingBox) -> \
        Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
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


def split_linestring_by_ratios(line: LineString, ratios: List[float]) -> List[LineString]:
    # TODO: Rifare (not working :-( )
    """
    Splits a LineString into segments based on provided ratios.

    Parameters:
    - line (LineString): Input line geometry.
    - ratios (List[float]): List of float ratios summing to 1.0.

    Returns:
    - List[LineString]: Segmented line parts.
    """
    if not isinstance(line, LineString):
        raise TypeError("Input must be a LineString.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Ratios must sum to 1.0.")

    total_length = line.length
    distances = np.cumsum(ratios[:-1]) * total_length  # Exclude last ratio

    split_points = [line.interpolate(d) for d in distances]

    segments = []
    current_line = line

    for pt in split_points:
        result = split(current_line, pt)
        parts = list(result.geoms)  # Get actual geometries

        # Sort by position along original line
        parts = sorted(parts, key=lambda seg: seg.project(Point(line.coords[0])))

        segments.append(parts[0])

        current_line = parts[1]

    segments.append(current_line)  # Add the final piece

    return segments


def plot_segments(segments, line_title="LineString Segments"):
    fig, ax = plt.subplots()
    colors = cm.get_cmap('tab10', len(segments))  # Use colormap with enough distinct colors

    for i, seg in enumerate(segments):
        x, y = seg.xy
        ax.plot(x, y, label=f'Segment {i + 1}', color=colors(i), linewidth=3)

        # Optional: add point labels at start of each segment
        ax.text(x[0], y[0], f'{i + 1}', fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    ax.set_title(line_title)
    ax.set_aspect('equal')
    ax.legend()
    plt.grid(True)
    plt.show()
def plot_linestring(linestring, color='blue', linewidth=2, show_points=False, point_color='red'):
    """
    Plots a Shapely LineString using matplotlib.

    Parameters:
    - linestring (shapely.geometry.LineString): The LineString to plot.
    - color (str): Color of the line.
    - linewidth (int): Width of the line.
    - show_points (bool): If True, show the vertices of the LineString.
    - point_color (str): Color of the vertices, if shown.
    """
    if not isinstance(linestring, LineString):
        raise TypeError("Input must be a shapely.geometry.LineString")

    x, y = linestring.xy
    plt.plot(x, y, color=color, linewidth=linewidth)

    if show_points:
        plt.plot(x, y, 'o', color=point_color)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Shapely LineString Plot")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def IoU(first_bbox: ImageBoundingBox, second_bbox: ImageBoundingBox) -> float:
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    Args:
        first_bbox (ImageBoundingBox): The first bounding box.
        second_bbox (ImageBoundingBox): The second bounding box.
    Returns:
        float: The IoU value, which is the ratio of the area of intersection to the area of union.
    """
    x_left = max(first_bbox.top_left_x, second_bbox.top_left_x)
    y_top = min(first_bbox.top_left_y, second_bbox.top_left_y)
    x_right = min(first_bbox.bottom_right_x, second_bbox.bottom_right_x)
    y_bottom = max(first_bbox.bottom_right_y, second_bbox.bottom_right_y)

    inter_width = max(0.0, x_right - x_left)
    inter_height = max(0.0, y_top - y_bottom)
    intersection = inter_width * inter_height

    area1 = (first_bbox.bottom_right_x - first_bbox.top_left_x) * (first_bbox.top_left_y - first_bbox.bottom_right_y)
    area2 = (second_bbox.bottom_right_x - second_bbox.top_left_x) * (
            second_bbox.top_left_y - second_bbox.bottom_right_y)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def distance_bbox_point(bbox: ImageBoundingBox, point_x: float, point_y: float) -> float:
    """
    Calculates the shortest distance from a point to a bounding box.
    Args:
        bbox (ImageBoundingBox): The bounding box.
        point (Tuple[float, float]): The point as a tuple of (x, y) coordinates.
    Returns:
        float: The shortest distance from the point to the bounding box.
    """
    x = max(bbox.top_left_x, min(point_x, bbox.bottom_right_x))
    y = max(bbox.bottom_right_y, min(point_y, bbox.top_left_y))
    return math.sqrt((x - point_x) ** 2 + (y - point_y) ** 2)
