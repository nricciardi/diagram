import unittest
import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.image.bbox.bbox2p import ImageBoundingBox2Points
from core.image.tensor_image import TensorImage
from core.image.image import Image
from src.extractor.flowchart.gnr_extractor import GNRFlowchartExtractor
from src.extractor.flowchart.multistage_extractor import ElementTextTypeOutcome, ArrowTextTypeOutcome


class TestGNRFlowchartExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = GNRFlowchartExtractor('test-extractor')

    def test_compute_text_associations(self):

        self.extractor.element_precedent_over_arrow_in_text_association = True

        element_bboxes = [
            ImageBoundingBox2Points(
                category="node1",
                box=torch.tensor([0, 0, 10, 10], dtype=torch.float32),
                trust=1
            ),
            ImageBoundingBox2Points(
                category="node2",
                box=torch.tensor([100, 100, 200, 200], dtype=torch.float32),
                trust=1
            ),
        ]

        arrow_bboxes = [
            ImageBoundingBox2Points(
                category="arrow1",
                box=torch.tensor([50, 50, 60, 60], dtype=torch.float32),
                trust=1
            ),
            ImageBoundingBox2Points(
                category="arrow2",
                box=torch.tensor([0, 0, 10, 10], dtype=torch.float32),
                trust=1
            ),
        ]

        text_bboxes = [
            ImageBoundingBox2Points(
                category="text-node1",
                box=torch.tensor([0, 0, 8, 8], dtype=torch.float32),
                trust=1
            ),
            ImageBoundingBox2Points(
                category="text-node2",
                box=torch.tensor([90, 90, 180, 180], dtype=torch.float32),
                trust=1
            ),
            ImageBoundingBox2Points(
                category="text-arrow1",
                box=torch.tensor([48, 48.5, 63, 63.5], dtype=torch.float32),
                trust=1
            ),
        ]

        element_text_associations, arrow_text_associations = self.extractor._compute_text_associations(
            "fake-diagram-id",
            element_bboxes,
            arrow_bboxes,
            text_bboxes
        )

        self.assertEqual(2, len(element_text_associations.keys()))
        self.assertEqual(1, len(arrow_text_associations.keys()))

        self.assertEqual(
            "text-node1",
            element_text_associations[element_bboxes[0]][0].category
        )

        self.assertEqual(
            "text-node2",
            element_text_associations[element_bboxes[1]][0].category
        )

        self.assertEqual(
            "text-arrow1",
            arrow_text_associations[arrow_bboxes[0]][0].category
        )

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

    def test_arrow_text_type(self):
        arrow_box: torch.Tensor = torch.tensor([5, 10, 8, 1])
        text_box: torch.Tensor = torch.tensor([7, 6, 12, 5])

        arrow_bbox: ImageBoundingBox2Points = ImageBoundingBox2Points(category='arrow', box=arrow_box, trust=1.0)
        text_bbox: ImageBoundingBox2Points = ImageBoundingBox2Points(category='text', box=text_box, trust=1.0)

        self.assertEqual(self.extractor._arrow_text_type(diagram_id='flowchart', arrow_bbox=arrow_bbox, text_bbox=text_bbox), ArrowTextTypeOutcome.MIDDLE)

    def test_digitalize_text(self):
        bbox_text: ImageBoundingBox2Points = ImageBoundingBox2Points("text", torch.Tensor([339, 339, 91, 84]), trust=0.9)
        image: Image = TensorImage.from_str("..\\..\\..\\dataset\\source\\fa\\test\\writer018_fa_001.png")
        self.assertEqual("q0", self.extractor._digitalize_text(diagram_id="flowchart", image=image, text_bbox=bbox_text))

if __name__ == '__main__':
    unittest.main()
