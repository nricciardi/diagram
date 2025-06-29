from enum import StrEnum
from dataclasses import dataclass, field
from typing import List

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""

class FlowchartElementCategory(StrEnum):
    TERMINAL = "TerminalNode"
    PROCESS = "ProcessNode"
    DECISION = "DecisionNode"
    INPUT_OUTPUT = "InputOutputNode"
    CIRCLE = "CircleNode"
    SUBROUTINE = "SubroutineNode"


@dataclass(frozen=True, slots=True)
class Element:
    """
    Represents a flowchart element with a category and associated text.
    Attributes:
        category (str): The category or type of the flowchart element.
        inner_text (List[str]): A list of strings representing the inner text of the element.
        outer_text (List[str]): A list of strings representing the outer text of the element.
    Methods:
        to_dict() -> dict:
            Converts the Element instance into a dictionary representation.
        from_dict(data: dict) -> 'Element':
            Populates the Element instance from a dictionary representation.
    """

    category: str
    inner_text: List[str] = field(default_factory=list)
    outer_text: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "inner_text": self.inner_text,
            "outer_text": self.outer_text,
        }

    def from_dict(self, data: dict) -> 'Element':
        object.__setattr__(self, "category", data["category"])
        object.__setattr__(self, "inner_text", data["inner_text"])
        object.__setattr__(self, "outer_text", data["outer_text"])
        return self