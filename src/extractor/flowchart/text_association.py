from dataclasses import dataclass

from core.image.bbox.bbox import ImageBoundingBox


@dataclass(frozen=True)
class TextAssociation:
    text_bbox: ImageBoundingBox
