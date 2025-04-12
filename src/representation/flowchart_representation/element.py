from enum import Enum
from dataclasses import dataclass
from typing import List

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""

class FlowchartElementCategory(Enum):
    TERMINAL: str = "TerminalNode"
    PROCESS: str = "ProcessNode"
    DECISION: str = "DecisionNode"
    INPUT_OUTPUT: str = "InputOutputNode"
    CIRCLE: str = "CircleNode"
    SUBROUTINE: str = "SubroutineNode"


@dataclass(frozen=True, slots=True)
class Element:
    """
    Flowchart element
    """

    category: str
    inner_text: List[str]
    outer_text: List[str]
