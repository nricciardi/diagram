from enum import Enum
from dataclasses import dataclass

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""


class FlowchartRelationCategory(Enum):
    ARROW: str = "Arrowhead"
    OPEN_LINK: str = "OpenLink"
    DOTTED_ARROW: str = "DottedArrowhead"


@dataclass(frozen=True, slots=True)
class Relation:
    """
    Flowchart relation
    """

    category: str
    source_id: str | None
    target_id: str | None
    label: str | None = None

