from typing import List, Type, override, Optional
import sys, os

from src.representation.flowchart_representation.element import FlowchartElementCategory, Element
from src.representation.flowchart_representation.relation import FlowchartRelationCategory, Relation

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer

class FlowchartToMermaidTransducer(Transducer):
    """
    Converts flowchart representations into Mermaid.js syntax.
    This transducer is specifically designed to handle flowchart diagrams and their elements, transforming them into
    a textual representation compatible with Mermaid.
    """
    

    def __init__(self, identifier: str):
        super().__init__(identifier)

    @override
    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    @override
    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [
            FlowchartRepresentation,

        ]

    @staticmethod
    def wrap_element(category: str, label: str) -> str:

        if label.strip() == "":
            label = category

        match category:
            case FlowchartElementCategory.CIRCLE.value:
                return f"(({label}))"
            case FlowchartElementCategory.TERMINAL.value:
                return f"([{label}])"
            case FlowchartElementCategory.PROCESS.value:
                return f"({label})"
            case FlowchartElementCategory.DECISION.value:
                return "{" + label + "}"
            case FlowchartElementCategory.INPUT_OUTPUT.value:
                return f"[/{label}/]"
            case FlowchartElementCategory.SUBROUTINE.value:
                return f"[{label}]"
            case _:
                raise ValueError(f"Unknown flowchart element category: {category}")

    @staticmethod
    def wrap_relation(category: str, label: str) -> str:
        match category:
            case FlowchartRelationCategory.ARROW.value:
                if label.strip() == "":
                    return "-->"
                return f"-->|{label}|"
            case FlowchartRelationCategory.OPEN_LINK.value:
                if label.strip() == "":
                    return " --- "
                return f"---|{label}|"
            case FlowchartRelationCategory.DOTTED_ARROW.value:
                if label.strip() == "":
                    return "-.->"
                return f" -. {label} .->"
            case _:
                raise ValueError(f"Unknown flowchart relation category: {category}")

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        """
        Transforms a diagram representation into a Mermaid flowchart representation.
        Args:
            diagram_id (str): The unique identifier for the diagram.
            diagram_representation (DiagramRepresentation): The representation of the diagram to be transduced.
                Must be an instance of `FlowchartRepresentation`.
        Returns:
            TransducerOutcome: An object containing the transduced diagram's ID, the Mermaid flowchart payload,
            and the markup language used ("mermaid").
        Raises:
            AssertionError: If `diagram_representation` is not an instance of `FlowchartRepresentation`.
        Notes:
            - The method constructs a Mermaid flowchart by iterating over the elements and relations
              in the `diagram_representation`.
            - Each element is wrapped with its category and inner text, while relations are formatted
              with their source, category, and target.
        """
        
        assert isinstance(diagram_representation, FlowchartRepresentation)

        body: str = "flowchart TD\n"
        for identifier, element in enumerate(diagram_representation.elements):
            element_text = ""
            if len(element.inner_text) > 0:
                element_text += f"{' '.join(element.inner_text)}\n"
            relation_text = ""
            if len(element.outer_text) > 0:
                relation_text += f"{' '.join(element.outer_text)}\n"

            element_text = element_text.strip()

            body += f"\t{identifier}{self.wrap_element(element.category, element_text)}\n"

        body += "\n"
        for relation in diagram_representation.relations:
            if relation.source_index is None or relation.target_index is None:
                continue
            body += f"\t{relation.source_index}"

            relation_text = ""
            if len(relation.source_text) > 0:
                relation_text += f"{' '.join(relation.source_text)}\n"
            if len(relation.middle_text) > 0:
                relation_text += f"{' '.join(relation.middle_text)}\n"
            if len(relation.inner_text) > 0:
                relation_text += f"{' '.join(relation.inner_text)}\n"
            if len(relation.target_text) > 0:
                relation_text += f"{' '.join(relation.target_text)}\n"

            relation_text = relation_text.strip()

            body += f"{self.wrap_relation(relation.category, relation_text)}{relation.target_index}\n"

        outcome: TransducerOutcome = TransducerOutcome(diagram_id=diagram_id, payload=body, markup_language="mermaid")
        return outcome
