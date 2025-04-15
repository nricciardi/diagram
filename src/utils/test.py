import unittest
import torch
from core.image.bbox.bbox2p import ImageBoundingBox2Points
from src.utils.bbox_utils import bbox_distance, bbox_overlap


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


if __name__ == '__main__':
    unittest.main()
