from typing import Dict, List
from dataclasses import dataclass
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.element import Element
from src.representation.flowchart_representation.relation import Relation


@dataclass
class FlowchartRepresentation(DiagramRepresentation):
    """
    Flowchart representation
    """

    elements: Dict[str, Element]
    relations: List[Relation]

    def dump(self, output_path: str):
        pass
    
    def load(self, input_path: str):
        pass
