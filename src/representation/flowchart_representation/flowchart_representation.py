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

    """
        The syntax is: first all elements - with id, category and label separated by ;; -
        then a new line and all relations - with category, source_id, target_id and label separeted by ;;
    """
    def dump(self, output_path: str):
        
        for identifier, element in self.elements.items():
            with open(output_path, "a") as file:
                file.write(f"{identifier};;{element.category};;{"" if element.label is None else element.label}\n")
                
        with open(output_path, "a") as file:
            file.write("\n")
            
        for relation in self.relations:
            with open(output_path, "a") as file:
                file.write(f"{relation.category};;{"" if relation.source_id is None else relation.source_id};;{"" if relation.target_id is None else relation.target_id};;{"" if relation.label is None else relation.label}\n")
        
    
    def load(self, input_path: str):
        
        with open(input_path, "r") as file:
            lines: list[str] = file.readlines()
        
        self.elements = {}
        self.relations = []
        break_line = len(lines)
        
        for idx, line in enumerate(lines):
            if line == "\n":
                break_line = idx
                break
            identifier, category, label = line.split(";;")
            label = label.strip()
            label = (None if label.strip() == "" else label.strip())
            self.elements[identifier] = Element(identifier, category, label)
            
        lines = lines[break_line + 1:]
        for line in lines:
            category, source_id, target_id, label = line.split(";;")
            source_id = (None if source_id == "" else source_id)
            target_id = (None if target_id == "" else target_id)
            label = label.strip()
            label = (None if label == "" else label)
            self.relations.append(Relation(category, source_id, target_id, label))

