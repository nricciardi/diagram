from typing import List, Type
from python_mermaid.diagram import MermaidDiagram, Link, Node
from python_mermaid.utils import snake_case

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer
from src.wellknown_markuplang import WellKnownMarkupLanguage

class FlowchartToMermaidTransducer(Transducer):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [
            FlowchartRepresentation,

        ]

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        assert isinstance(diagram_representation, FlowchartRepresentation)

        nodes: list[Node] = [
            Node(
                id=node_id,
                content=element.label,
                shape=element.category
            ) for node_id, element in diagram_representation.elements.items()
        ]

        links: list[Link] = [
            Link(
                origin=next((node for node in nodes if node.id == snake_case(relation.source_id)), None),
                end=next((node for node in nodes if node.id == snake_case(relation.target_id)), None),
                shape=relation.category,
                message=relation.label
            ) for relation in diagram_representation.relations
        ]

        links = [link for link in links if link.origin is not None and link.end is not None]

        body: MermaidDiagram = MermaidDiagram(
            title=diagram_id,
            nodes=nodes,
            links=links
        )
        
        outcome: TransducerOutcome = TransducerOutcome(diagram_id=diagram_id, payload=body.__str__(), markup_language=WellKnownMarkupLanguage.MERMAID.value)
        return outcome
