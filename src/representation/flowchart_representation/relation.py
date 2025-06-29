from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""


class FlowchartRelationCategory(Enum):
    ARROW: str = "Arrow"
    OPEN_LINK: str = "OpenLink"
    DOTTED_ARROW: str = "DottedArrow"


@dataclass(frozen=True, slots=True)
class Relation:
    """
    A class representing a relation in a flowchart representation.
    Attributes:
        category (str): The category of the relation.
        source_index (Optional[int]): The ID of the source node in the relation.
        target_index (Optional[int]): The ID of the target node in the relation.
        inner_text (List[str]): A list of strings representing the inner text of the relation.
        source_text (List[str]): A list of strings representing the text near the source node.
        target_text (List[str]): A list of strings representing the text near the target node.
        middle_text (List[str]): A list of strings representing the middle text of the relation.
    Methods:
        to_dict() -> dict:
            Converts the Relation object into a dictionary representation.
        from_dict(data: dict) -> 'Relation':
            Populates the Relation object from a dictionary representation.
        get_text() -> str:
            Retrieves the concatenated inner text of the relation.
    """
    

    category: str
    source_index: Optional[int] = field(default=None)
    target_index: Optional[int] = field(default=None)
    inner_text: List[str] = field(default_factory=list)
    source_text: List[str] = field(default_factory=list)
    target_text: List[str] = field(default_factory=list)
    middle_text: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "source_index": self.source_index,
            "target_index": self.target_index,
            "inner_text": self.inner_text,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "middle_text": self.middle_text,
        }

    def from_dict(self, data: dict) -> 'Relation':
        object.__setattr__(self, "category", data["category"])
        object.__setattr__(self, "source_index", data["source_index"])
        object.__setattr__(self, "target_index", data["target_index"])
        object.__setattr__(self, "inner_text", data["inner_text"])
        object.__setattr__(self, "source_text", data["source_text"])
        object.__setattr__(self, "target_text", data["target_text"])
        object.__setattr__(self, "middle_text", data["middle_text"])
        return self

