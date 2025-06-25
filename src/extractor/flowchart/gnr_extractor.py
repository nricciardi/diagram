import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, override
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from src.classifier.preprocessing.processor import GrayScaleProcessor, MultiProcessor
import torch
from shapely.geometry import Polygon, LineString
from torchvision.models.detection import FasterRCNN
from core.image.bbox.bbox import ImageBoundingBox
from core.image.bbox.bbox2p import ImageBoundingBox2Points
from core.image.image import Image
from src.extractor.arrow.arrow import Arrow
from src.extractor.bbox_detection.target import FlowchartElementCategoryIndex, Lookup
from src.extractor.flowchart.multistage_extractor import MultistageFlowchartExtractor, ArrowTextTypeOutcome, \
    ElementTextTypeOutcome, ObjectRelation
from src.utils.bbox_utils import bbox_overlap, \
    distance_bbox_point, split_linestring_by_ratios, bbox_vertices, crop_image
from src.wellknown_diagram import WellKnownDiagram
from src.extractor.text_extraction.text_extractor import TextExtractor

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GNRFlowchartExtractor(MultistageFlowchartExtractor):

    text_digitizer: TextExtractor
    bbox_detector: FasterRCNN

    preprocessor = MultiProcessor([
        GrayScaleProcessor()
    ])

    # TODO: change name in CLI
    element_precedent_over_arrow_in_text_association: bool
    element_text_overlap_threshold: float
    element_text_distance_threshold: float
    arrow_text_discard_distance_threshold: float
    arrow_text_inner_distance_threshold: float
    arrow_crop_delta_size_x: float
    arrow_crop_delta_size_y: float
    element_arrow_overlap_threshold: float
    element_arrow_distance_threshold: float

    ratios = [0.2, 0.6, 0.2]  # Source, Middle, Target
    bbox_trust_thresholds: Dict[int, Optional[float]] = field(default_factory=dict)

    def __post_init__(self):
        for key, threshold in self.bbox_trust_thresholds.items():
            if threshold > 1 or threshold < 0:
                raise ValueError(f"bbox_trust_thresholds (category: {key}) must be between 0 and 1")


    @override
    def update_thresholds(self, diagram_id: str, image: Image) -> None:
        pass
        # TODO: not working
        # tensor: torch.Tensor = image.as_tensor()
        # if len(tensor.shape) == 2:
        #     longest_side: float = max(tensor.shape[0], tensor.shape[1])
        # elif len(tensor.shape) == 3:
        #     longest_side: float = max(image.as_tensor().shape[0], image.as_tensor().shape[1], image.as_tensor().shape[2])
        # else:
        #     logger.warning(f"Unexpected tensor shape {tensor.shape}, assuming last two dimensions are height and width\n")
        #     longest_side: float = max(tensor.shape[-2], tensor.shape[-1])
        # self.element_arrow_distance_threshold = 0.2 * longest_side
        # self.arrow_text_discard_distance_threshold = 0.2 * longest_side
        # self.arrow_text_inner_distance_threshold = 0.2 * longest_side

    def to_device(self, device: str):
        self.bbox_detector = self.bbox_detector.to(device)
        self.text_digitizer.to_device(device)

    def get_device(self) -> str:
        assert next(self.bbox_detector.parameters()).device.type == self.text_digitizer.get_device()

        return self.text_digitizer.get_device()

    @override
    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.GRAPH_DIAGRAM.value,
            WellKnownDiagram.FLOW_CHART.value,
        ]

    @override
    def _is_arrow_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW.value]

    @override
    def _is_element_category(self, diagram_id: str, category: str) -> bool:
        return category not in [Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW.value],
                                Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW_TAIL.value],
                                Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW_HEAD.value],
                                Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.TEXT.value]]

    @override
    def _is_text_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.TEXT.value]

    @override
    def _is_arrow_head_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW_HEAD.value]

    @override
    def _is_arrow_tail_category(self, diagram_id: str, category: str) -> bool:
        return category == Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW_TAIL.value]

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
                element_text_distance: float = text_bbox.distance(element_bbox)

                if element_text_distance < minimum_element_text_distance:
                    minimum_element_text_distance = element_text_distance
                    minimum_element_text_bbox = element_bbox

                # distance >= 0 -> if 0, stop searching
                if minimum_element_text_distance == 0:
                    break

            minimum_arrow_text_distance: float = float('inf')
            minimum_distance_arrow_to_text: Optional[Arrow] = None
            for arrow in arrows:
                arrow_text_distance: float = arrow.distance_to_bbox(text_bbox)

                if arrow_text_distance < minimum_arrow_text_distance:
                    minimum_arrow_text_distance = arrow_text_distance
                    minimum_distance_arrow_to_text = arrow

            if minimum_arrow_text_distance < minimum_element_text_distance or (
                    minimum_arrow_text_distance == minimum_element_text_distance and not self.element_precedent_over_arrow_in_text_association):
                arrow_text_associations[minimum_distance_arrow_to_text].append(text_bbox)

            else:
                element_text_associations[minimum_element_text_bbox].append(text_bbox)

        tot_element_text_associations = sum([len(v) for k, v in element_text_associations.items()])
        tot_arrow_text_associations = sum([len(v) for k, v in arrow_text_associations.items()])
        # TODO fix
        logger.debug(f"{tot_element_text_associations} element-text associations found")
        logger.debug(f"{tot_arrow_text_associations} arrow-text associations found")
        # TODO fix
        assert tot_element_text_associations + tot_arrow_text_associations == len(text_bboxes)
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
                    category=Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW.value],
                    arrow=arrow,
                    source_index=source,
                    target_index=target,
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
            distance = element_bbox.distance(text_bbox)
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
    def _manage_wrong_computed_arrows(self, diagram_id: str, image: Image, arrow_bboxes: List[ImageBoundingBox]) -> \
    List[Optional[Arrow]]:

        """
        Args:
            diagram_id:
            image: tensor (C, H, W)
            arrow_bboxes: (x1, y1, x2, y2)

        Returns:

        """
        managed_arrows: List[Optional[Arrow]] = []
        crop_size: Tuple[int, int, int] = (1, image.as_tensor().shape[1] + int(self.arrow_crop_delta_size_y) + 1, image.as_tensor().shape[2] + int(self.arrow_crop_delta_size_x) + 1)
        _, H, W = image.as_tensor().shape
        for arrow_bbox in arrow_bboxes:
            x1, y1, x2, y2 = int(arrow_bbox.bottom_left_x), int(arrow_bbox.top_left_y), int(arrow_bbox.bottom_right_x), int(arrow_bbox.bottom_left_y)
            crop_bbox: torch.Tensor = torch.ones(crop_size).to(self.get_device())
            crop_center_x: int = int(crop_bbox.shape[2] // 2)
            crop_center_y: int = int(crop_bbox.shape[1] // 2)
            delta_x: int = int((x2 - x1) // 2) + int(self.arrow_crop_delta_size_x // 2)
            assert crop_center_x - delta_x > 0 and crop_center_x + delta_x < crop_size[2]
            delta_y: int = int((y2 - y1) // 2) + int(self.arrow_crop_delta_size_y // 2)
            assert crop_center_y - delta_y > 0 and crop_center_y + delta_y < crop_size[1]
            orig_center_x: int = (x2 - x1) // 2 + x1
            orig_center_y: int = (y2 - y1) // 2 + y1

            delta_y_left: int = delta_y
            delta_y_right: int = delta_y

            delta_x_left: int = delta_x
            delta_x_right: int = delta_x

            orig_y_start: int = max(0, orig_center_y - delta_y)
            if orig_y_start == 0:
                delta_y_left -= delta_y - orig_center_y
            orig_y_end: int = min(H, orig_center_y + delta_y)
            if orig_y_end == H:
                delta_y_right -= orig_center_y + delta_y - H
            orig_x_start: int = max(0, orig_center_x - delta_x)
            if orig_x_start == 0:
                delta_x_left -= delta_x - orig_center_x
            orig_x_end: int = min(W, orig_center_x + delta_x)
            if orig_x_end == W:
                delta_x_right -= orig_center_x + delta_x - W

            crop_bbox[:, (crop_center_y - delta_y_left):(crop_center_y + delta_y_right), (crop_center_x - delta_x_left):(crop_center_x + delta_x_right)] = \
                image.as_tensor()[:, orig_y_start:orig_y_end, orig_x_start:orig_x_end]
            prediction = self.bbox_detector(crop_bbox.unsqueeze(0))[0]
            head: Optional[torch.Tensor] = None
            tail: Optional[torch.Tensor] = None
            head_score: float = 0.0
            tail_score: float = 0.0
            for bbox, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if label.item() == FlowchartElementCategoryIndex.ARROW_HEAD.value and score.item() > head_score:
                    head = bbox
                    head_score = score.item()
                if label.item() == FlowchartElementCategoryIndex.ARROW_TAIL.value and score.item() > tail_score:
                    tail = bbox
                    tail_score = score.item()

            if head is not None and tail is not None:
                managed_arrows.append(Arrow.from_bboxes(
                head_bbox=ImageBoundingBox2Points.from_image(category=Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW_HEAD.value],
                                                box=head, trust=head_score, image=image),
                tail_bbox=ImageBoundingBox2Points.from_image(category=Lookup.table_target_int_to_str_by_diagram_id[diagram_id][FlowchartElementCategoryIndex.ARROW_TAIL.value],
                                                  box=tail, trust=tail_score, image=image)
                                                  ))
            else:
                managed_arrows.append(None)

        return managed_arrows

    @override
    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        image_tensor = image.as_tensor().unsqueeze(0).float() / 255.0      # unsqueeze(0) to fake a batch: (C=1, H, W) -> (1, C=1, H, W)

        prediction = self.bbox_detector(image_tensor)[0]
        bboxes: List[ImageBoundingBox] = []


        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if label.item() not in Lookup.table_target_int_to_str_by_diagram_id[diagram_id]:
                continue    # this should not be recognized here

            bboxes.append(ImageBoundingBox2Points.from_image(category=Lookup.table_target_int_to_str_by_diagram_id[diagram_id][label.item()], box=box, trust=score.item(), image=image))

        if logging.root.level <= 10: # TODO disable
            # Draw predictions
            img_cpu = image_tensor.squeeze(0).cpu()
            boxes = prediction['boxes']
            labels = prediction['labels']
            drawn = draw_bounding_boxes(img_cpu, boxes=boxes, labels=[str(l.item()) for l in labels], width=2)
            plt.imshow(to_pil_image(drawn))
            plt.axis('off')
            plt.show()

        return bboxes


    @override
    def _filter_bboxes(self, diagram_id: str, bboxes: List[ImageBoundingBox]) -> List[ImageBoundingBox]:
        """
        Filter bboxes based on trust threshold, i.e. keep only bbox which has trust > threshold
        """

        filtered_bboxes = []

        for bbox in bboxes:
            if bbox.category not in self.bbox_trust_thresholds or \
                    self.bbox_trust_thresholds[Lookup.table_str_to_target_int_by_diagram_id[diagram_id][bbox.category]] is None or \
                    bbox.trust > self.bbox_trust_thresholds[Lookup.table_str_to_target_int_by_diagram_id[diagram_id][bbox.category]]:

                filtered_bboxes.append(bbox)

        return filtered_bboxes
