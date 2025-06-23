import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, override
from src.classifier.preprocessing.processor import GrayScaleProcessor, MultiProcessor
import torch
from shapely.geometry import Polygon, LineString
from torchvision.models.detection import FasterRCNN

from core.image.bbox.bbox import ImageBoundingBox
from core.image.bbox.bbox2p import ImageBoundingBox2Points
from core.image.image import Image
from src.extractor.arrow.arrow import Arrow
from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, \
    ElementTextTypeOutcome, ObjectRelation
from src.flowchart_element_category import FlowchartElementCategoryIndex, Lookup
from src.utils.bbox_utils import bbox_overlap, bbox_distance, bbox_vertices, \
    distance_bbox_point, split_linestring_by_ratios
from src.wellknown_diagram import WellKnownDiagram
from src.extractor.text_extraction.text_extractor import TrOCRTextExtractorSmall, TextExtractor

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GNRFlowchartExtractor(MultistageFlowchartExtractor):

    text_digitizer: TextExtractor
    bbox_detector: FasterRCNN

    preprocessor = MultiProcessor([
        GrayScaleProcessor()
    ])
    element_precedent_over_arrow_in_text_association: bool = True
    element_text_overlap_threshold: float = 0.5  # TODO find optimal threshold
    element_text_distance_threshold: float = 10  # TODO find optimal threshold
    arrow_text_discard_distance_threshold: float = 10  # TODO find optimal threshold
    arrow_text_inner_distance_threshold: float = 2 # TODO find optimal threshold
    element_arrow_overlap_threshold: float = 0.1  # TODO find optimal threshold
    element_arrow_distance_threshold: float = 20.  # TODO find optimal threshold
    ratios = [0.2, 0.6, 0.2]  # Source, Middle, Target

    def to_device(self, device: str):
        self.bbox_detector = self.bbox_detector.to(device)
        self.text_digitizer.to_device(device)

    @override
    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.GRAPH_DIAGRAM.value,
            WellKnownDiagram.FLOW_CHART.value,
        ]

    @override
    def _is_arrow_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table[FlowchartElementCategoryIndex.ARROW.value]

    @override
    def _is_element_category(self, diagram_id: str, category: str) -> bool:
        return category not in [Lookup.table[FlowchartElementCategoryIndex.ARROW.value],
                                Lookup.table[FlowchartElementCategoryIndex.ARROW_TAIL.value],
                                Lookup.table[FlowchartElementCategoryIndex.ARROW_HEAD.value],
                                Lookup.table[FlowchartElementCategoryIndex.TEXT.value]]

    @override
    def _is_text_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table[FlowchartElementCategoryIndex.TEXT.value]

    @override
    def _is_arrow_head_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table[FlowchartElementCategoryIndex.ARROW_HEAD.value]

    @override
    def _is_arrow_tail_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table[FlowchartElementCategoryIndex.ARROW_TAIL.value]

    def _preprocess(self, diagram_id: str, image: Image) -> Image:

        image = self.preprocessor.process(image)
        return image

    def _compute_text_associations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox],
                                   arrows: List[Arrow], text_bboxes: List[ImageBoundingBox])\
            -> Tuple[Dict[ImageBoundingBox, List[ImageBoundingBox]], Dict[Arrow, List[ImageBoundingBox]]]:

        element_text_associations: Dict[ImageBoundingBox, List[ImageBoundingBox]] = defaultdict(list)
        arrow_text_associations: Dict[Arrow, List[ImageBoundingBox]] = defaultdict(list)

        for text_bbox in text_bboxes:

            minimum_element_text_distance: float = float('inf')
            minimum_element_text_bbox: Optional[ImageBoundingBox] = None
            for element_bbox in element_bboxes:
                element_text_distance: float = bbox_distance(text_bbox, element_bbox)

                if element_text_distance < minimum_element_text_distance:
                    minimum_element_text_distance = element_text_distance
                    minimum_element_text_bbox = element_bbox

            minimum_arrow_text_distance: float = float('inf')
            minimum_distance_arrow_to_text: Optional[Arrow] = None
            for arrow in arrows:
                arrow_text_distance: float = bbox_distance(text_bbox, arrow.bbox)

                if arrow_text_distance < minimum_arrow_text_distance:
                    minimum_arrow_text_distance = arrow_text_distance
                    minimum_distance_arrow_to_text = arrow

            if minimum_arrow_text_distance < minimum_element_text_distance or (
                    minimum_arrow_text_distance == minimum_element_text_distance and not self.element_precedent_over_arrow_in_text_association):
                arrow_text_associations[minimum_distance_arrow_to_text].append(text_bbox)

            else:
                element_text_associations[minimum_element_text_bbox].append(text_bbox)

        return element_text_associations, arrow_text_associations

    def _digitalize_text(self, diagram_id: str, image: Image, text_bbox: ImageBoundingBox) -> str:
        """
        Extracts and digitalizes text from a specified region of an image.
        Args:
            diagram_id (str): The identifier of the diagram being processed.
            image (Image): The input image containing the diagram.
            text_bbox (ImageBoundingBox): The bounding box specifying the region of the image 
                where the text is located.
        Returns:
            str: The extracted and digitalized text from the specified region of the image.
        Notes:
            - The method processes the image tensor to crop the region defined by the bounding box.
            - The cropped region is converted to a PIL image and passed through a text recognition 
              model to extract the text.
            - The extracted text is stripped of leading and trailing whitespace before being returned.
        """

        logger.debug("Analyzing the text found...")
        generated_text: str = self.text_digitizer.extract_text(image, text_bbox)
        logger.debug(f"Text found is '{generated_text}'")

        return generated_text.strip()

    def _compute_relations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox],
                           arrows: List[Arrow]) -> List[ObjectRelation]:
        """
            Computes the relationships between elements and arrows in a diagram based on their bounding boxes.
            Args:
                diagram_id (str): The identifier of the diagram being processed.
                element_bboxes (List[ImageBoundingBox]): A list of bounding boxes representing the elements in the diagram.
                arrows (List[Arrow]): A list of arrows in the diagram.
            Returns:
                List[ObjectRelation]: A list of object relations, where each relation specifies the category of the arrow
                                        and the indices of the source and target elements it connects.
        """

        ret: List[ObjectRelation] = []

        for arrow in arrows:

            target = source = None
            distance_head_min = distance_tail_min = None
            for i, element in enumerate(element_bboxes):
                element_bbox: ImageBoundingBox = element
                distance_element_head: float = distance_bbox_point(bbox=element_bbox, point_x=arrow.x_head, point_y=arrow.y_head)
                if distance_element_head < self.element_arrow_distance_threshold and (distance_head_min is None or distance_element_head < distance_head_min):
                    target = i
                    distance_head_min = distance_element_head

                distance_element_tail: float = distance_bbox_point(bbox=element_bbox, point_x=arrow.x_tail, point_y=arrow.y_tail)
                if distance_element_tail < self.element_arrow_distance_threshold and (distance_tail_min is None or distance_element_tail < distance_tail_min):
                    source = i
                    distance_tail_min = distance_element_tail

            ret.append(
                ObjectRelation(
                    category=Lookup.table[FlowchartElementCategoryIndex.ARROW.value],
                    source=source,
                    target=target,
                )
            )

        return ret

    def _element_text_type(self, diagram_id: str, element_bbox: ImageBoundingBox,
                           text_bbox: ImageBoundingBox) -> ElementTextTypeOutcome:

        """
        Computes the relation between a text and an element
        Args:
            diagram_id (str): The identifier of the diagram being processed
            element_bbox (ImageBoundingBox): The bbox (2 points) of the element associated to the text
            text_bbox (ImageBoundingBox): The bbox (2 points) of the text associated to the element
        Returns:
            ElementTextTypeOutcome:
            Position of the text with respect to the element (INNER, OUTER) or no relation (DISCARD)
        Notes:
            - The function assumes that then there is a relation if there is an overlap between the bboxes or the
            distance is lower than a certain threshold
            - Only if there is no overlap, the distance is taken into account
            - The bboxes are assumed to be 2 points
        """

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

    def _arrow_text_type(self, diagram_id: str, arrow: Arrow, text_bbox: ImageBoundingBox) -> ArrowTextTypeOutcome:
        """
        Computes the relation between an arrow and a text

        Args:
            diagram_id (str): The identifier of the diagram being processed
            arrow (Arrow): The bbox of the arrow associated to the text
            text_bbox (ImageBoundingBox): The bbox of the text associated to the element
        Returns:
            ArrowTextTypeOutcome:
            Position of the text with respect to the arrow (INNER, SOURCE, MIDDLE, TARGET) or no relation (DISCARD)
        Notes:
            - While it is technically possible to pass 2 points bboxes as parameters, to have more accurate results,
            it is suggested to use 4 points bboxes
            - The function assumes that there is a relation if there is overlap between the bboxes
            or the distance is lower than a certain threshold
        """

        logger.debug('Computing vertices arrow-text...')
        arrow_bbox_vertices, text_bbox_vertices = bbox_vertices(bbox1=arrow.bbox, bbox2=text_bbox)
        arrow_line = LineString(coordinates=[[arrow.x_tail, arrow.y_tail], [arrow.x_head, arrow.y_head]])
        text_poly = Polygon(text_bbox_vertices)

        logger.debug('Computing distance arrow-text...')
        distance = arrow_line.distance(text_poly)
        logger.debug(f'Distance arrow-text is {distance}')

        outcome: ArrowTextTypeOutcome = ArrowTextTypeOutcome.INNER

        if distance > self.arrow_text_discard_distance_threshold:
            outcome = ArrowTextTypeOutcome.DISCARD

        logger.debug('Computing arrow splits...')
        splits: List[LineString] = split_linestring_by_ratios(line=arrow_line, ratios=self.ratios)
        source: LineString = splits[0]
        middle: LineString = splits[1]
        target: LineString = splits[2]

        if self.arrow_text_inner_distance_threshold < distance <= self.arrow_text_discard_distance_threshold:
            logger.debug('Computing distance splits-text...')
            min_distance = source.distance(text_poly)
            outcome = ArrowTextTypeOutcome.SOURCE
            middle_distance = middle.distance(text_poly)
            if min_distance > middle_distance:
                min_distance = middle_distance
                outcome = ArrowTextTypeOutcome.MIDDLE
            target_distance = target.distance(text_poly)
            if min_distance > target_distance:
                outcome = ArrowTextTypeOutcome.TARGET

        logger.debug(f'Outcome {outcome} for position arrow-text')
        return outcome

    @override
    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        image = image.as_tensor().unsqueeze(0).unsqueeze(0).float() / 255.0
        prediction = self.bbox_detector(image)
        bboxes: List[ImageBoundingBox] = []

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            bboxes.append(ImageBoundingBox2Points(Lookup.table[label.item()], box, score.item()))

        return bboxes
