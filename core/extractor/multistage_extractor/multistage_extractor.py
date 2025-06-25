import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from core.extractor.extractor import Extractor
from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image
from core.representation.representation import DiagramRepresentation

logger = logging.getLogger(__name__)


@dataclass
class MultiStageExtractor(Extractor, ABC):

    def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:

        self.update_thresholds(diagram_id, image)

        image = self._preprocess(diagram_id, image)

        bboxes: List[ImageBoundingBox] = self._extract_diagram_objects(diagram_id, image)

        logger.debug(f"{len(bboxes)} extracted")

        bboxes = self._filter_bboxes(diagram_id, bboxes)

        return self._build_diagram_representation(diagram_id, image, bboxes)

    def update_thresholds(self, diagram_id: str, image: Image) -> None:
        """

        Args:
            diagram_id:
            image:

        Returns:

        """

    def _preprocess(self, diagram_id: str, image: Image) -> Image:
        """
        Hook method to preprocess image based on diagram
        """

    @abstractmethod
    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        """
        Extract diagram's objects as bounding boxes
        """

    def _filter_bboxes(self, diagram_id: str, bboxes: List[ImageBoundingBox]) -> List[ImageBoundingBox]:
        """
        Filter bboxes based on trust threshold, i.e. keep only bbox which has trust > threshold
        """

        return bboxes

    @abstractmethod
    def _build_diagram_representation(self, diagram_id: str, image: Image, bboxes: List[ImageBoundingBox]) -> DiagramRepresentation:
        """
        Build diagram representation based on diagram type, image and related bounding boxes
        """







