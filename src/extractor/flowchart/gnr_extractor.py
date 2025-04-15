from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image

from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, \
    ElementTextTypeOutcome, ObjectRelation
from src.utils.bbox_distance import bbox_distance
from src.utils.bbox_overlap import bbox_overlap
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

    def _compute_text_associations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox],
                                   arrow_bboxes: List[ImageBoundingBox],
                                   text_bboxes: List[ImageBoundingBox]) -> Tuple[
        Dict[ImageBoundingBox, List[ImageBoundingBox]], Dict[ImageBoundingBox, List[ImageBoundingBox]]]:
        pass

    def _digitalize_text(self, diagram_id: str, image: Image, text_bbox: ImageBoundingBox) -> str:
        pass

    def _compute_relations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox],
                           arrow_bboxes: List[ImageBoundingBox]) -> List[ObjectRelation]:
        pass

    def _element_text_type(self, diagram_id: str, element_bbox: ImageBoundingBox,
                           text_bbox: ImageBoundingBox) -> ElementTextTypeOutcome:

        overlap_text: float = bbox_overlap(bbox1=element_bbox, bbox2=text_bbox)

        distance: float = 0
        if overlap_text == 0:
            distance = bbox_distance(bbox1=element_bbox, bbox2=text_bbox)

        overlap_threshold: float = 0.5  # TODO find optimal threshold
        distance_threshold: float = 10  # TODO find optimal threshold
        outcome: ElementTextTypeOutcome = ElementTextTypeOutcome.INNER

        if overlap_text >= overlap_threshold:
            outcome = ElementTextTypeOutcome.INNER
        if 0 < overlap_text < overlap_threshold or (overlap_text == 0 and distance <= distance_threshold):
            outcome = ElementTextTypeOutcome.OUTER
        if overlap_text == 0 and distance > distance_threshold:
            outcome = ElementTextTypeOutcome.DISCARD

        return outcome

    def _arrow_text_type(self, diagram_id: str, arrow_bbox: ImageBoundingBox,
                         text_bbox: ImageBoundingBox) -> ArrowTextTypeOutcome:
        pass

    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        pass
