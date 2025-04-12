from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass
from core.extractor.multistage_extractor.multistage_extractor import MultiStageExtractor
from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.element import Element
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.representation.flowchart_representation.relation import Relation


@dataclass(frozen=True)
class ObjectRelation:
    category: str
    source_index: Optional[int]
    target_index: Optional[int]


class ElementTextTypeOutcome(Enum):
    INNER = "inner"
    OUTER = "outer"
    DISCARD = "discard"


class ArrowTextTypeOutcome(Enum):
    INNER = "inner"
    SOURCE = "source"
    MIDDLE = "middle"
    TARGET = "target"
    DISCARD = "discard"


@dataclass
class MultistageFlowchartExtractor(MultiStageExtractor, ABC):

    parallelization: bool = False

    def _build_diagram_representation(self, diagram_id: str, image: Image, bboxes: List[ImageBoundingBox]) -> FlowchartRepresentation:
        if self.parallelization:
            return self.__par_build_diagram_representation(diagram_id, image, bboxes)
        else:
            return self.__seq_build_diagram_representation(diagram_id, image, bboxes)


    def __seq_build_diagram_representation(self, diagram_id: str, image: Image, bboxes: List[ImageBoundingBox]) -> FlowchartRepresentation:

        element_bboxes: List[ImageBoundingBox] = [bbox for bbox in bboxes if self._is_element_category(diagram_id, bbox.category)]
        arrow_bboxes: List[ImageBoundingBox] = [bbox for bbox in bboxes if self._is_arrow_category(diagram_id, bbox.category)]
        text_bboxes: List[ImageBoundingBox] = [bbox for bbox in bboxes if self._is_text_category(diagram_id, bbox.category)]

        elements_texts_associations, arrows_texts_associations = self._compute_text_associations(diagram_id, element_bboxes, arrow_bboxes, text_bboxes)

        objects_relations: List[ObjectRelation] = self._compute_relations(diagram_id, element_bboxes, arrow_bboxes)

        elements: List[Element] = self._build_elements(diagram_id, image, elements_texts_associations)

        relations: List[Relation] = self._build_relations(diagram_id, image, objects_relations, arrows_texts_associations)

        return FlowchartRepresentation(
            elements=elements,
            relations=relations
        )


    def __par_build_diagram_representation(self, diagram_id: str, image: Image, bboxes: List[ImageBoundingBox]) -> FlowchartRepresentation:
        raise NotImplemented()      # TODO


    @abstractmethod
    def _is_arrow_category(self, diagram_id: str, category: str) -> bool:
        raise NotImplemented()

    @abstractmethod
    def _is_element_category(self, diagram_id: str, category: str) -> bool:
        raise NotImplemented()

    @abstractmethod
    def _is_text_category(self, diagram_id: str, category: str) -> bool:
        raise NotImplemented()

    @abstractmethod
    def _compute_text_associations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox], arrow_bboxes: List[ImageBoundingBox],
                                   text_bboxes: List[ImageBoundingBox]) -> Tuple[Dict[ImageBoundingBox, List[ImageBoundingBox]], Dict[ImageBoundingBox, List[ImageBoundingBox]]]:
        """
        For each text bbox, associate non-text object having the minimum distance

        :return: (elements associations, arrows associations)
        """

    @abstractmethod
    def _digitalize_text(self, diagram_id: str, image: Image, text_bbox: ImageBoundingBox) -> str:
        """
        Digitalize text:

        1. Extract image patch from original one
        2. Digitalize text in that region

        :return: digitalized text
        """

    @abstractmethod
    def _compute_relations(self, diagram_id: str, element_bboxes: List[ImageBoundingBox], arrow_bboxes: List[ImageBoundingBox]) -> List[ObjectRelation]:
        """
        Compute relations between elements and arrows

        source_index and target_index must be indices of element_bboxes lists
        """

    @abstractmethod
    def _element_text_type(self, diagram_id: str, element_bbox: ImageBoundingBox, text_bbox: ImageBoundingBox) -> ElementTextTypeOutcome:
        """
        Return if text of element is inner, outer or must be discarded
        """

    @abstractmethod
    def _arrow_text_type(self, diagram_id: str, arrow_bbox: ImageBoundingBox, text_bbox: ImageBoundingBox) -> ArrowTextTypeOutcome:
        """
        Return if text of arrow is inner, outer or must be discarded
        """

    def _build_elements(self, diagram_id: str, image: Image, elements_texts_associations: Dict[ImageBoundingBox, List[ImageBoundingBox]]) -> List[Element]:

        elements: List[Element] = []
        for element_bbox, associated_text_bboxes in elements_texts_associations.items():
            inner_text: List[str] = []
            outer_text: List[str] = []

            for associated_text_bbox in associated_text_bboxes:
                outcome = self._element_text_type(diagram_id, element_bbox, associated_text_bbox)
                text = self._digitalize_text(diagram_id, image, associated_text_bbox)

                match outcome:
                    case ElementTextTypeOutcome.INNER:
                        inner_text.append(text)
                    case ElementTextTypeOutcome.OUTER:
                        outer_text.append(text)
                    case ElementTextTypeOutcome.DISCARD:
                        raise NotImplemented()  # TODO: secchio degli scarti

            element: Element = Element(element_bbox.category, inner_text, outer_text)
            elements.append(element)

        return elements


    def _build_relations(self, diagram_id: str, image: Image, objects_relations: List[ObjectRelation], arrow_texts_associations: Dict[ImageBoundingBox, List[ImageBoundingBox]]) -> List[Relation]:
        relations: List[Relation] = []

        for obj_relation in objects_relations:
            inner_text: List[str] = []
            source_text: List[str] = []
            middle_text: List[str] = []
            target_text: List[str] = []

            for arrow_bbox, associated_text_bboxes in arrow_texts_associations.items():
                for associated_text_bbox in associated_text_bboxes:
                    outcome = self._element_text_type(diagram_id, arrow_bbox, associated_text_bbox)
                    text = self._digitalize_text(diagram_id, image, associated_text_bbox)

                    match outcome:
                        case ArrowTextTypeOutcome.INNER:
                            inner_text.append(text)
                        case ArrowTextTypeOutcome.SOURCE:
                            source_text.append(text)
                        case ArrowTextTypeOutcome.MIDDLE:
                            middle_text.append(text)
                        case ArrowTextTypeOutcome.TARGET:
                            target_text.append(text)
                        case ElementTextTypeOutcome.DISCARD:
                            raise NotImplemented()  # TODO: secchio degli scarti

            relation = Relation(
                category=obj_relation.category,
                source_id=obj_relation.source_index,
                target_id=obj_relation.target_index,
                inner_text=inner_text,
                source_text=source_text,
                middle_text=middle_text,
                target_text=target_text,
            )

            relations.append(relation)

        return relations