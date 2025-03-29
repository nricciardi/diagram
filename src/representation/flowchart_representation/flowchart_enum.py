from enum import Enum

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""

"""
    NOTA:
           Da parlarne insieme magari. Ci sono le annotazioni: il collegamento (la linea) se la includiamo
            deve essere una relazione (AnnotationLink) mentre il testo deve essere un'entità ("AnnotationNode").
            Giusto?
"""
class FlowchartElementEnum(Enum):
    TERMINAL: str = "TerminalNode"
    PROCESS: str = "ProcessNode"
    DECISION: str = "DecisionNode"
    INPUT_OUTPUT: str = "InputOutputNode"
    ANNOTATION: str = "AnnotationNode"
    SUBROUTINE: str = "SubroutineNode"
    
class FlowchartRelationEnum(Enum):
    ARROW: str = "Arrowhead"
    ANNOTATION_LINK: str = "AnnotationLink"