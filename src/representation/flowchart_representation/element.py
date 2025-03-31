from enum import Enum
from dataclasses import dataclass

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""

class FlowchartElementCategory(Enum):
    PROCESS: str = "normal"
    TERMINAL: str = "round-edge"
    DECISION: str = "rhombus"
    INPUT_OUTPUT: str = "parallelogram"
    CIRCLE: str = "circle"
    SUBROUTINE: str = "subroutine-shape"


@dataclass(frozen=True, slots=True)
class Element:
    """
    Flowchart element
    """

    identifier: str
    category: str
    label: str | None = None
