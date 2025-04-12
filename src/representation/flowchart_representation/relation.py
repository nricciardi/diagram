from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

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
    source_id: Optional[str]
    target_id: Optional[str]
    inner_text: List[str]
    source_text: List[str]
    target_text: List[str]
    middle_text: List[str]
