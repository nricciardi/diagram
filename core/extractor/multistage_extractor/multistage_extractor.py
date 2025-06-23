import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from core.extractor.extractor import Extractor
from core.image.bbox.bbox import ImageBoundingBox
from core.image.image import Image
from core.representation.representation import DiagramRepresentation

logger = logging.getLogger(__name__)


@dataclass
class MultiStageExtractor(Extractor, ABC):

    bbox_trust_threshold: Optional[float] = None

    def __post_init__(self):
        if self.bbox_trust_threshold is not None:
            if self.bbox_trust_threshold > 1 or self.bbox_trust_threshold < 0:
                raise ValueError("bbox_trust_threshold must be between 0 and 1")

    def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:
        image = self._preprocess(diagram_id, image)

        bboxes = self._extract_diagram_objects(diagram_id, image)

        logger.debug(f"{len(bboxes)} extracted")

        if self.bbox_trust_threshold is not None:
            bboxes = self._filter_bboxes(bboxes)

        return self._build_diagram_representation(diagram_id, image, bboxes)

    def _preprocess(self, diagram_id: str, image: Image) -> Image:
        """
        Hook method to preprocess image based on diagram
        """

    @abstractmethod
    def _extract_diagram_objects(self, diagram_id: str, image: Image) -> List[ImageBoundingBox]:
        """
        Extract diagram's objects as bounding boxes
        """

    def _filter_bboxes(self, bboxes: List[ImageBoundingBox]) -> List[ImageBoundingBox]:
        """
        Filter bboxes based on trust threshold, i.e. keep only bbox which has trust > threshold
        """

        return [bbox for bbox in bboxes if bbox.trust > self.bbox_trust_threshold]


    @abstractmethod
    def _build_diagram_representation(self, diagram_id: str, image: Image, bboxes: List[ImageBoundingBox]) -> DiagramRepresentation:
        """
        Build diagram representation based on diagram type, image and related bounding boxes
        """







