import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict
from shapely.geometry import Polygon

from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image

from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, \
    ElementTextTypeOutcome, ObjectRelation
from src.utils.bbox_utils import bbox_overlap, bbox_distance, bbox_vertices, bbox_split

from src.wellknown_diagram import WellKnownDiagram

logger = logging.getLogger(__name__)


@dataclass
class GNRFlowchartExtractor(MultistageFlowchartExtractor):
    element_text_overlap_threshold: float = 0.5  # TODO find optimal threshold
    element_text_distance_threshold: float = 10  # TODO find optimal threshold
    arrow_text_distance_threshold: float = 10  # TODO find optimal threshold

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

        logger.debug('Computing overlap element-text...')
        overlap_text: float = bbox_overlap(bbox1=element_bbox, bbox2=text_bbox)
        logger.debug(f'Overlap element-text is {overlap_text}')

        distance: float = 0
        if overlap_text == 0:
            logger.debug('Computing distance element-text...')
            distance = bbox_distance(bbox1=element_bbox, bbox2=text_bbox)
            logger.debug(f'Distance element-text is {distance}')

        outcome: ElementTextTypeOutcome = ElementTextTypeOutcome.INNER

        if overlap_text >= self.element_text_overlap_threshold:
            outcome = ElementTextTypeOutcome.INNER
        if 0 < overlap_text < self.element_text_overlap_threshold or (
                overlap_text == 0 and distance <= self.element_text_distance_threshold):
            outcome = ElementTextTypeOutcome.OUTER
        if overlap_text == 0 and distance > self.element_text_distance_threshold:
            outcome = ElementTextTypeOutcome.DISCARD

        logger.debug(f'Outcome {outcome.value} for overlapping element-text')
        return outcome

    def _arrow_text_type(self, diagram_id: str, arrow_bbox: ImageBoundingBox,
                         text_bbox: ImageBoundingBox) -> ArrowTextTypeOutcome:

        logger.debug('Computing vertices arrow-text...')
        arrow_bbox_vertices, text_bbox_vertices = bbox_vertices(bbox1=arrow_bbox, bbox2=text_bbox)
        arrow_poly = Polygon(arrow_bbox_vertices)
        text_poly = Polygon(text_bbox_vertices)
        logger.debug('Computing distance arrow-text...')
        distance = arrow_poly.distance(text_poly)
        logger.debug(f'Distance arrow-text is {distance}')

        outcome: ArrowTextTypeOutcome = ArrowTextTypeOutcome.INNER
        if distance > self.arrow_text_distance_threshold:
            outcome = ArrowTextTypeOutcome.DISCARD
        if distance == 0:
            outcome = ArrowTextTypeOutcome.INNER

        if 0 < distance < self.arrow_text_distance_threshold:
            direction: str = ...  # 'top-bottom' or 'left-right'
            ratios = [20, 60, 20]
            logger.debug('Computing arrow split...')
            source, middle, target = bbox_split(bbox=arrow_bbox, direction=direction, ratios=ratios)  # TODO split
            source_poly = Polygon(source)
            middle_poly = Polygon(middle)
            target_poly = Polygon(target)
            min_distance = source_poly.distance(text_poly)
            outcome = ArrowTextTypeOutcome.SOURCE
            middle_distance = middle_poly.distance(text_poly)
            if min_distance > middle_distance:
                min_distance = middle_distance
                outcome = ArrowTextTypeOutcome.MIDDLE
            target_distance = target_poly.distance(text_poly)
            if min_distance > target_distance:
                min_distance = target_distance
                outcome = ArrowTextTypeOutcome.TARGET

        logger.debug(f'Outcome {outcome} for overlapping arrow-text')
        return outcome

    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        pass
