from typing import Dict
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_enum import FlowchartElementEnum, FlowchartRelationEnum

class FlowchartElementId():
    """
    Flowchart element id
    
    NOTA: L'ho messo nel caso in cui serva qualcos'altro... che so: un check sull'ID; non si voglia l'id stringa ma numerico, etc.
    """
    def __init__(self, id: str):
        self.id = id

class FlowchartElement():
    """
    Flowchart element
    """
    def __init__(self, id: FlowchartElementId, category: FlowchartElementEnum, label: str = ""):
        self.id = id
        self.category = category
        self.label = label
        
    def __str__(self):
        return f"{self.category.name} ({self.label})"

    def __repr__(self):
        return self.__str__()
    
class FlowchartRelation():
    """
    Flowchart relation
    """
    def __init__(self, category: FlowchartRelationEnum, source_id: FlowchartElementId | None, target_id: FlowchartElementId | None, label: str = ""):
        self.category = category
        self.source_id = source_id
        self.target_id = target_id
        self.label = label
        
    def __str__(self):
        return f"{self.category.name}({self.source_id} -> {self.target_id}) ({self.label})"

    def __repr__(self):
        return self.__str__()
    
class FlowchartRepresentation(DiagramRepresentation):
    """
    Flowchart representation
    """
    def __init__(self, elements: Dict[FlowchartElementId, FlowchartElement], relations: list[FlowchartRelation]):
        super().__init__()
        self.elements = elements
        self.relations = relations
    
    def dump(self, output_path: str):
        pass
    
    def load(self, input_path: str):
        pass
