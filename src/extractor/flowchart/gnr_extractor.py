import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon

from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image
from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, \
    ElementTextTypeOutcome, ObjectRelation
from src.utils.bbox_utils import bbox_overlap, bbox_distance, bbox_vertices, bbox_split, bbox_relative_position
from src.wellknown_diagram import WellKnownDiagram

from torchvision.transforms.functional import to_pil_image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


@dataclass
class GNRFlowchartExtractor(MultistageFlowchartExtractor):
    element_precedent_over_arrow_in_text_association: bool = True
    element_text_overlap_threshold: float = 0.5  # TODO find optimal threshold
    element_text_distance_threshold: float = 10  # TODO find optimal threshold
    arrow_text_distance_threshold: float = 10  # TODO find optimal threshold
    element_arrow_overlap_threshold: float = 0.1  # TODO find optimal threshold
    element_arrow_distance_threshold: float = 20.  # TODO find optimal threshold
    ratios = [0.2, 0.6, 0.2]  # Source, Middle, Target
    
    # For text digitalization
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

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
        tensor = image.as_tensor()  # [C, H, W], torch.Tensor

        # Get bounding box as integers
        left = int(min(text_bbox.top_left_x, text_bbox.bottom_left_x))
        right = int(max(text_bbox.top_right_x, text_bbox.bottom_right_x))
        top = int(min(text_bbox.top_left_y, text_bbox.top_right_y))
        bottom = int(max(text_bbox.bottom_left_y, text_bbox.bottom_right_y))

        _, H, W = tensor.shape
        left = max(0, left)
        right = min(W, right)
        top = max(0, top)
        bottom = min(H, bottom)

        cropped_tensor = tensor[:, top:bottom, left:right]
        if cropped_tensor.ndim == 2:
            cropped_tensor = cropped_tensor.unsqueeze(0).repeat(3, 1, 1)
        elif cropped_tensor.shape[0] == 1:
            cropped_tensor = cropped_tensor.repeat(3, 1, 1)
        cropped_image = to_pil_image(cropped_tensor)

        logger.debug("Analyzing the text found...")
        pixel_values = self.processor(images=cropped_image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text: str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.debug(f"Text found is '{generated_text}'")

        return generated_text.strip()

    def _compute_relations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox],
                           arrow_bboxes: List[ImageBoundingBox]) -> List[ObjectRelation]:
        ret: List[ObjectRelation] = []
        """
        self.element_arrow_distance_threshold
        """
        for arrow in arrow_bboxes:
            overlaps = []
            # TODO: Distance bucket
            elements_direction = "horizontally" # TODO: "horizontally" or "vertically"
            arrow_pointing = "right" # TODO: "right", "left", "up" or "down"
            for idx, elem in enumerate(element_bboxes):
                overlap_score = bbox_overlap(bbox1=elem, bbox2=arrow)
                if overlap_score > self.element_arrow_overlap_threshold:
                    relative_position = bbox_relative_position(first_bbox=arrow, second_bbox=elem, direction=elements_direction)
                    overlaps.append((idx, overlap_score, relative_position))
                else:
                    pass

            overlaps.sort(key=lambda x: x[1], reverse=True)

            # If we have at least two overlaps AND we have at least one of each overlap type (e.g. up/down or left/right)
            if len(overlaps) >= 2 and len(set([overlap[2] for overlap in overlaps])) >= 2:
                if elements_direction == "horizontally":
                    source_id = list(filter((lambda element : element[2] == "left" if arrow_pointing == "right" else "right"), overlaps))[0][0]
                    target_id = list(filter((lambda element : element[2] == "right" if arrow_pointing == "right" else "left"), overlaps))[0][0]
                elif elements_direction == "vertically":
                    source_id = list(filter((lambda element : element[2] == "up" if arrow_pointing == "down" else "down"), overlaps))[0][0]
                    target_id = list(filter((lambda element : element[2] == "down" if arrow_pointing == "down" else "up"), overlaps))[0][0]
                else:
                    logger.warning("elements_direction doesn't have a viable type; ignoring for now")
                    continue
            elif len(set([overlap[2] for overlap in overlaps])) == 1:
                pass
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
        logger.debug('Computing overlap percentage arrow-text...')
        intersection = arrow_poly.intersection(text_poly)
        inter_area = intersection.area
        text_area = text_poly.area
        overlap_text = inter_area / text_area
        logger.debug(f'Overlap percentage arrow-text is {overlap_text}')

        logger.debug('Computing distance arrow-text...')
        distance = arrow_poly.distance(text_poly)
        logger.debug(f'Distance arrow-text is {distance}')

        outcome: ArrowTextTypeOutcome = ArrowTextTypeOutcome.INNER
        if distance > self.arrow_text_distance_threshold:
            outcome = ArrowTextTypeOutcome.DISCARD
        if distance == 0 and overlap_text == 1:
            outcome = ArrowTextTypeOutcome.INNER

        direction: str = 'vertically'  # TODO 'vertically' or 'horizontally'
        arrow_head: str = 'down'  # TODO 'up', 'down', 'left', or 'right'
        logger.debug('Computing arrow splits...')
        splits = bbox_split(bbox=arrow_bbox, direction=direction, ratios=self.ratios, arrow_head=arrow_head)
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
                max_inter_area = inter_area_target
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
                min_distance = target_distance
                outcome = ArrowTextTypeOutcome.TARGET

        logger.debug(f'Outcome {outcome} for overlapping arrow-text')
        return outcome

    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        pass
