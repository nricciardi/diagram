from enum import Enum

"""
For reference: https://en.wikipedia.org/wiki/Flowchart#Building_blocks
"""

class FlowchartElementEnum(Enum):
    TERMINAL: str = "TerminalNode"
    PROCESS: str = "ProcessNode"
    DECISION: str = "DecisionNode"
    INPUT_OUTPUT: str = "InputOutputNode"
    CIRCLE: str = "CircleNode"
    SUBROUTINE: str = "SubroutineNode"
    
class FlowchartRelationEnum(Enum):
    ARROW: str = "Arrowhead"
    OPEN_LINK: str = "OpenLink"
    DOTTED_ARROW: str = "DottedArrowhead"