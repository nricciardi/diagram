import unittest
import torch

from core.image.bbox.bbox2p import ImageBoundingBox2Points
from src.extractor.flowchart.gnr_extractor import GNRFlowchartExtractor
from src.extractor.flowchart.multistage_extractor import ElementTextTypeOutcome


class TestGNRFlowchartExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = GNRFlowchartExtractor('test extractor')

    def test_element_text_type(self):
        box1: torch.Tensor = torch.tensor([2, 20, 6, 15], dtype=torch.float64)
        bbox1: ImageBoundingBox2Points = ImageBoundingBox2Points(category='node', box=box1, trust=1.0)
        box2: torch.Tensor = torch.tensor([10, 12, 17, 8])
        bbox2: ImageBoundingBox2Points = ImageBoundingBox2Points(category='text', box=box2, trust=1.0)
        box3: torch.Tensor = torch.tensor([2, 11, 6, 7], dtype=torch.float64)
        bbox3: ImageBoundingBox2Points = ImageBoundingBox2Points(category='node', box=box3, trust=1.0)
        box4: torch.Tensor = torch.tensor([3, 9, 5, 5])
        bbox4: ImageBoundingBox2Points = ImageBoundingBox2Points(category='text', box=box4, trust=1.0)
        self.assertEqual(self.extractor._element_text_type(diagram_id='flowchart', element_bbox=bbox1, text_bbox=bbox2), ElementTextTypeOutcome.OUTER)
        self.assertEqual(self.extractor._element_text_type(diagram_id='flowchart', element_bbox=bbox3, text_bbox=bbox4), ElementTextTypeOutcome.INNER)


if __name__ == '__main__':
    unittest.main()
