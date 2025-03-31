from enum import Enum
from dataclasses import dataclass

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""


class FlowchartRelationCategory(Enum):
    ARROW: str = "normal"
    DOTTED_ARROW: str = "dotted"


@dataclass(frozen=True, slots=True)
class Relation:
    """
    Flowchart relation
    """

    category: str
    source_id: str | None
    target_id: str | None
    label: str | None = None

