from typing import List, Type

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.representation.flowchart_representation.element import FlowchartElementCategory, Element
from src.representation.flowchart_representation.relation import FlowchartRelationCategory, Relation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer
from src.wellknown_markuplang import WellKnownMarkupLanguage

class FlowchartToD2Transducer(Transducer):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [FlowchartRepresentation]

    @staticmethod
    def wrap_element(category: str, label: str, elem_id: str) -> str:
        match category:
            case FlowchartElementCategory.CIRCLE.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: circle\n"
            case FlowchartElementCategory.TERMINAL.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: oval\n"
            case FlowchartElementCategory.PROCESS.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.style.border-radius: 8\n"
            case FlowchartElementCategory.DECISION.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: diamond\n"
            case FlowchartElementCategory.INPUT_OUTPUT.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: parallelogram\n"
            case FlowchartElementCategory.SUBROUTINE.value:
                return f"{elem_id}: {label}\n"
            case _:
                raise ValueError(f"Unknown flowchart element category: {category}")

    @staticmethod
    def wrap_relation(category: str, label: str, target_id: str) -> str:
        match category:
            case FlowchartRelationCategory.ARROW.value:
                if label == "":
                    return f"->{target_id}\n"
                return f"->{target_id}: {label}\n"
            case FlowchartRelationCategory.OPEN_LINK.value:
                if label == "":
                    return f"--{target_id}\n"
                return f"--{target_id}: {label}\n"
            case FlowchartRelationCategory.DOTTED_ARROW.value:
                if label == "":
                    return f"->{target_id} {{\n" \
                           "\tstyle: {\n" \
                           "\tstroke-dash: 3\n" \
                           "\t}\n" \
                           "}\n"
                return f"->{target_id}: {label} {{\n" \
                       "\tstyle: {\n" \
                       "\t\tstroke-dash: 3\n" \
                       "\t}\n" \
                       "}\n"
            case _:
                raise ValueError(f"Unknown flowchart relation category: {category}")

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        assert isinstance(diagram_representation, FlowchartRepresentation)

        body: str = ""
        for identifier, element in enumerate(diagram_representation.elements):
            # NOTA: L'outer_text non è assolutamente utilizzato
            body += self.wrap_element(element.category, " ".join(element.inner_text), str(identifier))

        body += "\n"
        for relation in diagram_representation.relations:
            body += f"{relation.source_id}"
            body += self.wrap_relation(relation.category, "" if relation.get_text() is None else relation.get_text(), relation.target_id)

        outcome: TransducerOutcome = TransducerOutcome(diagram_id, WellKnownMarkupLanguage.D2_LANG.value, body)
        return outcome