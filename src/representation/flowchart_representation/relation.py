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
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "inner_text": self.inner_text,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "middle_text": self.middle_text,
        }

    def from_dict(self, data: dict) -> 'Relation':
        object.__setattr__(self, "category", data["category"])
        object.__setattr__(self, "source_id", data["source_id"])
        object.__setattr__(self, "target_id", data["target_id"])
        object.__setattr__(self, "inner_text", data["inner_text"])
        object.__setattr__(self, "source_text", data["source_text"])
        object.__setattr__(self, "target_text", data["target_text"])
        object.__setattr__(self, "middle_text", data["middle_text"])
        return self