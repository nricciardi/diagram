from dataclasses import dataclass
from typing import List, Tuple, Dict

from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image

from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, ElementTextTypeOutcome, ObjectRelation
from src.wellknown_diagram import WellKnownDiagram


@dataclass
class GNRFlowchartExtractor(MultistageFlowchartExtractor):

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.GRAPH_DIAGRAM.value,
            WellKnownDiagram.FLOW_CHART.value,
        ]

    def _is_arrow_category(self, diagram_id: str, category: str) -> bool:
        pass

    def _is_element_category(self, diagram_id: str, category: str) -> bool:
        pass

    def _is_text_category(self, diagram_id: str, category: str) -> bool:
        pass

    def _preprocess(self, diagram_id: str, image: Image) -> Image:
        pass

    def _compute_text_associations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox], arrow_bboxes: List[ImageBoundingBox],
                                   text_bboxes: List[ImageBoundingBox]) -> Tuple[Dict[ImageBoundingBox, List[ImageBoundingBox]], Dict[ImageBoundingBox, List[ImageBoundingBox]]]:
        pass

    def _digitalize_text(self, diagram_id: str, image: Image, text_bbox: ImageBoundingBox) -> str:
        pass

    def _compute_relations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox], arrow_bboxes: List[ImageBoundingBox]) -> List[ObjectRelation]:
        pass

    def _element_text_type(self, diagram_id: str, element_bbox: ImageBoundingBox, text_bbox: ImageBoundingBox) -> ElementTextTypeOutcome:
        pass

    def _arrow_text_type(self, diagram_id: str, arrow_bbox: ImageBoundingBox, text_bbox: ImageBoundingBox) -> ArrowTextTypeOutcome:
        pass

    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        pass
