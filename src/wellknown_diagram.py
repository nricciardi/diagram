from enum import Enum

class WellKnownDiagram(Enum):
    GRAPH_DIAGRAM = "graph"
    FLOW_CHART = "flowchart"
    OTHER = "other"

    @staticmethod
    def from_string(value: str):
        value = value.lower()
        for diagram in WellKnownDiagram:
            if diagram.value == value:
                return diagram
        return WellKnownDiagram.OTHER