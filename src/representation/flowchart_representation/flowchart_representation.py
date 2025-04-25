import json
from typing import Dict, List
from dataclasses import dataclass
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.element import Element
from src.representation.flowchart_representation.relation import Relation


@dataclass(frozen=True)
class FlowchartRepresentation(DiagramRepresentation):
    """
    A class representing a flowchart diagram, which consists of elements and relations.
    Attributes:
        elements (List[Element]): A list of elements in the flowchart.
        relations (List[Relation]): A list of relations between the elements in the flowchart.
    Methods:
        dump(output_path: str):
            Serializes the flowchart representation to a JSON file at the specified output path.
        load(input_path: str) -> 'FlowchartRepresentation':
            Loads a flowchart representation from a JSON file at the specified input path.
    """
    

    elements: List[Element]
    relations: List[Relation]


    def dump(self, output_path: str):

        with open(output_path, 'w') as file:
            json.dump({
            "elements": [element.to_dict() for element in self.elements],
            "relations": [relation.to_dict() for relation in self.relations],
            }, file, indent=4)

    @staticmethod
    def load(input_path: str) -> 'FlowchartRepresentation':
        
        with open(input_path, 'r') as file:
            data = json.load(file)
            return FlowchartRepresentation(elements=[Element("", [], []).from_dict(element) for element in data["elements"]],
                                           relations=[Relation("", "", "", [], [], [], []).from_dict(relation) for relation in data["relations"]])