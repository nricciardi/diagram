import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, override

from shapely.geometry import Polygon
from torchvision.models.detection import FasterRCNN

from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image
from src.extractor.arrow.arrow import Arrow
from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, \
    ElementTextTypeOutcome, ObjectRelation
from src.flowchart_element_category import FlowchartElementCategory
from src.utils.bbox_utils import bbox_overlap, bbox_distance, bbox_vertices, bbox_split, bbox_relative_position
from src.wellknown_diagram import WellKnownDiagram
from src.extractor.text_extraction.text_extractor import TrOCRTextExtractorSmall

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GNRFlowchartExtractor(MultistageFlowchartExtractor):

    bbox_detector: FasterRCNN

    element_precedent_over_arrow_in_text_association: bool = True
    element_text_overlap_threshold: float = 0.5  # TODO find optimal threshold
    element_text_distance_threshold: float = 10  # TODO find optimal threshold
    arrow_text_distance_threshold: float = 10  # TODO find optimal threshold
    element_arrow_overlap_threshold: float = 0.1  # TODO find optimal threshold
    element_arrow_distance_threshold: float = 20.  # TODO find optimal threshold
    ratios = [0.2, 0.6, 0.2]  # Source, Middle, Target

    text_digitizer = TrOCRTextExtractorSmall()

    @override
    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.GRAPH_DIAGRAM.value,
            WellKnownDiagram.FLOW_CHART.value,
        ]

    @override
    def _is_arrow_category(self, diagram_id: str, category: int) -> bool:
        return category == FlowchartElementCategory.ARROW.value

    @override
    def _is_element_category(self, diagram_id: str, category: int) -> bool:
        pass        # TODO

    @override
    def _is_text_category(self, diagram_id: str, category: int) -> bool:
        return category == FlowchartElementCategory.TEXT.value

    @override
    def _is_arrow_head_category(self, diagram_id: str, category: int) -> bool:
        return category == FlowchartElementCategory.ARROW_HEAD.value

    @override
    def _is_arrow_tail_category(self, diagram_id: str, category: int) -> bool:
        return category == FlowchartElementCategory.ARROW_TAIL.value

    def _preprocess(self, diagram_id: str, image: Image) -> Image:
        pass

    def _compute_text_associations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox],
                                   arrow_bboxes: List[ImageBoundingBox],
                                   text_bboxes: List[ImageBoundingBox])\
            -> Tuple[Dict[ImageBoundingBox, List[ImageBoundingBox]], Dict[ImageBoundingBox, List[ImageBoundingBox]]]:

        """
        All text bboxes are assigned to the nearest element bbox or arrow bbox, ignoring distance itself.
        In other words, distance may be also huge. After, pruning is supposed.

        Args:
            diagram_id (str): The identifier of the diagram being processed
            element_bboxes (List[ImageBoundingBox]): The bounding boxes of the elements
            arrow_bboxes (List[ImageBoundingBox]): The bounding boxes of the arrows
            text_bboxes (List[ImageBoundingBox]): The bounding boxes of the texts
        Returns:
            Tuple[Dict[ImageBoundingBox, List[ImageBoundingBox]], Dict[ImageBoundingBox, List[ImageBoundingBox]]]:

            Two dictionaries (A, B):

            - A: each element bbox (key) is mapped with its list of texts bboxes (value)
            - B: each arrows bbox (key) is mapped with its list of texts bboxes (value)
        """

        element_text_associations: Dict[ImageBoundingBox, List[ImageBoundingBox]] = defaultdict(list)
        arrow_text_associations: Dict[ImageBoundingBox, List[ImageBoundingBox]] = defaultdict(list)

        for text_bbox in text_bboxes:

            minimum_element_text_distance: float = float('inf')
            minimum_element_text_bbox: Optional[ImageBoundingBox] = None
            for element_bbox in element_bboxes:
                element_text_distance: float = bbox_distance(text_bbox, element_bbox)

                if element_text_distance < minimum_element_text_distance:
                    minimum_element_text_distance = element_text_distance
                    minimum_element_text_bbox = element_bbox

            minimum_arrow_text_distance: float = float('inf')
            minimum_arrow_text_bbox: Optional[ImageBoundingBox] = None
            for arrow_bbox in arrow_bboxes:
                arrow_text_distance: float = bbox_distance(text_bbox, arrow_bbox)

                if arrow_text_distance < minimum_arrow_text_distance:
                    minimum_arrow_text_distance = arrow_text_distance
                    minimum_arrow_text_bbox = arrow_bbox

            if minimum_arrow_text_distance < minimum_element_text_distance or (
                    minimum_arrow_text_distance == minimum_element_text_distance and not self.element_precedent_over_arrow_in_text_association):
                arrow_text_associations[minimum_arrow_text_bbox].append(text_bbox)

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
                           arrow_bboxes: List[Arrow]) -> List[ObjectRelation]:
        """
            Computes the relationships between elements and arrows in a diagram based on their bounding boxes.
            Args:
                diagram_id (str): The identifier of the diagram being processed.
                element_bboxes (List[ImageBoundingBox]): A list of bounding boxes representing the elements in the diagram.
                arrow_bboxes (List[ImageBoundingBox]): A list of bounding boxes representing the arrows in the diagram.
            Returns:
                ist[ObjectRelation]: A list of object relations, where each relation specifies the category of the arrow
                                        and the indices of the source and target elements it connects.
            Notes:
                - The function calculates overlaps and distances between arrows and elements to determine relationships.
                - Arrows are assumed to point in one of four directions: "right", "left", "up", or "down".
                - Relationships are determined based on overlap scores, distance thresholds, and relative positions.
                - If multiple overlaps exist, the function prioritizes overlaps with distinct relative positions.
                - If only one overlap exists, the source and target indices may point to the same element.
        """

        # TODO: si passa da arrow_bbox a 2 punti a testa e coda della freccia
        # TODO: non ci sarà più right left, bisogna ottenere testa e coda di ogni freccia e

        ret: List[ObjectRelation] = []
        for arrow in arrow_bboxes:
            overlaps = []
            arrow_pointing = "right"  # TODO: "right", "left", "up" or "down"
            for idx, elem in enumerate(element_bboxes):
                overlap_score = bbox_overlap(bbox1=elem, bbox2=arrow)
                if overlap_score > self.element_arrow_overlap_threshold:
                    relative_position = bbox_relative_position(first_bbox=arrow, second_bbox=elem)
                    overlaps.append((idx, overlap_score, relative_position))
                distance = bbox_distance(bbox1=elem, bbox2=arrow)
                if self.element_arrow_distance_threshold > distance:
                    relative_position = bbox_relative_position(first_bbox=arrow, second_bbox=elem)
                    distance_score = (distance / self.element_arrow_distance_threshold) + 1
                    overlaps.append((idx, distance_score, relative_position))
                    pass

            overlaps.sort(key=lambda x: x[1], reverse=True)
            source_id = target_id = None

            # If we have at least two overlaps AND we have at least one of each overlap type (e.g. up/down or left/right)
            if len(overlaps) >= 2 and len(set([overlap[2] for overlap in overlaps])) >= 2:
                if arrow_pointing == "right" or "left":
                    source_id = list(
                        filter((lambda element: element[2] == "left" if arrow_pointing == "right" else "right"),
                               overlaps))[0][0]
                    target_id = list(
                        filter((lambda element: element[2] == "right" if arrow_pointing == "right" else "left"),
                               overlaps))[0][0]
                else:
                    source_id = list(
                        filter((lambda element: element[2] == "up" if arrow_pointing == "down" else "down"), overlaps))[
                        0][0]
                    target_id = list(
                        filter((lambda element: element[2] == "down" if arrow_pointing == "down" else "up"), overlaps))[
                        0][0]
            elif len(set([overlap[2] for overlap in overlaps])) == 1:
                element = overlaps[0]
                if arrow_pointing == "right" or arrow_pointing == "left":
                    source_id = element[0] if element[2] == ("left" if arrow_pointing == "right" else "right") else None
                    target_id = element[0] if element[2] == ("right" if arrow_pointing == "right" else "left") else None
                elif arrow_pointing == "up" or arrow_pointing == "down":
                    source_id = element[0] if element[2] == ("up" if arrow_pointing == "down" else "down") else None
                    target_id = element[0] if element[2] == ("down" if arrow_pointing == "down" else "up") else None
            elif len(overlaps) == 1:
                source_id = overlaps[0][0]
                target_id = overlaps[0][0]
            else:
                continue

            ret.append(ObjectRelation(
                category=arrow.category,
                source_index=source_id,
                target_index=target_id
            ))

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
            arrow_bbox (ImageBoundingBox): The bbox of the arrow associated to the text
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

        # TODO: si passa da arrow_bbox a 2 punti a testa e coda della freccia

        logger.debug('Computing vertices arrow-text...')
        arrow_bbox_vertices, text_bbox_vertices = bbox_vertices(bbox1=arrow_bbox, bbox2=text_bbox)
        arrow_poly = Polygon(arrow_bbox_vertices)
        text_poly = Polygon(text_bbox_vertices)
        logger.debug('Computing overlap percentage arrow-text...')
        intersection = arrow_poly.intersection(text_poly)
        inter_area = intersection.area
        text_area = text_poly.area
        overlap_text = inter_area / text_area
        logger.debug(f'Overlap percentage arrow-text is {overlap_text}')

        logger.debug('Computing distance arrow-text...')
        distance = arrow_poly.distance(text_poly)  # distance = 0 if overlap_text > 0
        logger.debug(f'Distance arrow-text is {distance}')

        outcome: ArrowTextTypeOutcome = ArrowTextTypeOutcome.INNER

        if distance > self.arrow_text_distance_threshold:
            outcome = ArrowTextTypeOutcome.DISCARD
        if distance == 0 and overlap_text == 1:
            outcome = ArrowTextTypeOutcome.INNER

        arrow_head: str = 'down'  # TODO 'up', 'down', 'left', or 'right'
        logger.debug('Computing arrow splits...')
        splits = bbox_split(bbox=arrow_bbox, ratios=self.ratios, arrow_head=arrow_head)
        source, middle, target = splits[0], splits[1], splits[2]
        source_poly = Polygon(source)
        middle_poly = Polygon(middle)
        target_poly = Polygon(target)

        if distance == 0 and overlap_text < 1:
            logger.debug('Computing intersection splits-text...')
            intersection_source = source_poly.intersection(text_poly)
            inter_area_source = intersection_source.area
            max_inter_area = inter_area_source
            outcome = ArrowTextTypeOutcome.SOURCE
            intersection_middle = middle_poly.intersection(text_poly)
            inter_area_middle = intersection_middle.area
            if max_inter_area < inter_area_middle:
                max_inter_area = inter_area_middle
                outcome = ArrowTextTypeOutcome.MIDDLE
            intersection_target = target_poly.intersection(text_poly)
            inter_area_target = intersection_target.area
            if max_inter_area < inter_area_target:
                outcome = ArrowTextTypeOutcome.TARGET

        if 0 < distance <= self.arrow_text_distance_threshold:
            logger.debug('Computing distance splits-text...')
            min_distance = source_poly.distance(text_poly)
            outcome = ArrowTextTypeOutcome.SOURCE
            middle_distance = middle_poly.distance(text_poly)
            if min_distance > middle_distance:
                min_distance = middle_distance
                outcome = ArrowTextTypeOutcome.MIDDLE
            target_distance = target_poly.distance(text_poly)
            if min_distance > target_distance:
                outcome = ArrowTextTypeOutcome.TARGET

        logger.debug(f'Outcome {outcome} for overlapping arrow-text')
        return outcome

    @override
    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        pass    # TODO: model.predict() -> tensors -> List[ImageBoundingBox] ([nic] -> List, List, List)
