import unittest
from typing import List, Tuple

import torch
from core.image.bbox.bbox2p import ImageBoundingBox2Points
from src.utils.bbox_utils import bbox_distance, bbox_overlap, bbox_split


class TestUtils(unittest.TestCase):
    def test_bbox_distance(self):
        box1: torch.Tensor = torch.tensor([2, 20, 6, 15], dtype=torch.float64)
        bbox1: ImageBoundingBox2Points = ImageBoundingBox2Points(category='node', box=box1, trust=1.0)
        box2: torch.Tensor = torch.tensor([10, 12, 17, 8])
        bbox2: ImageBoundingBox2Points = ImageBoundingBox2Points(category='text', box=box2, trust=1.0)
        self.assertEqual(bbox_distance(bbox1=bbox1, bbox2=bbox2), 5)

    def test_bbox_overlap(self):
        box1: torch.Tensor = torch.tensor([2, 11, 6, 7], dtype=torch.float64)
        bbox1: ImageBoundingBox2Points = ImageBoundingBox2Points(category='node', box=box1, trust=1.0)
        box2: torch.Tensor = torch.tensor([3, 9, 5, 5])
        bbox2: ImageBoundingBox2Points = ImageBoundingBox2Points(category='text', box=box2, trust=1.0)
        self.assertEqual(bbox_overlap(bbox1=bbox1, bbox2=bbox2), 0.5)

    def test_bbox_split(self):
        expected_source: List[Tuple[float, float]] = [(5, 10), (8, 10), (8, 8.2), (5, 8.2)]
        expected_middle: List[Tuple[float, float]] = [(5, 8.2), (8, 8.2), (8, 2.8), (5, 2.8)]
        expected_target: List[Tuple[float, float]] = [(5, 2.8), (8, 2.8), (8, 1), (5, 1)]

        box1: torch.Tensor = torch.tensor([5, 10, 8, 1], dtype=torch.float64)
        bbox1: ImageBoundingBox2Points = ImageBoundingBox2Points(category='node', box=box1, trust=1.0)
        splits = bbox_split(bbox=bbox1, direction='height', ratios=[0.2, 0.6, 0.2], arrow_head='down')
        self.assertEqual(expected_target, splits[2])
        self.assertEqual(expected_middle, splits[1])
        self.assertEqual(expected_source, splits[0])



if __name__ == '__main__':
    unittest.main()
