from typing import List, Type, override, Optional

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

    @override
    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    @override
    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [FlowchartRepresentation]

    @staticmethod
    def wrap_element(category: str, label: str, elem_id: int) -> str:

        s = str(elem_id)
        if label.strip() != "":
            s += f": {label}\n"

        match category:
            case FlowchartElementCategory.CIRCLE.value:
                s += f"{elem_id}.shape: circle\n"
            case FlowchartElementCategory.TERMINAL.value:
                s += f"{elem_id}.shape: oval\n"
            case FlowchartElementCategory.PROCESS.value:
                s += f"{elem_id}.style.border-radius: 8\n"
            case FlowchartElementCategory.DECISION.value:
                s += f"{elem_id}.shape: diamond\n"
            case FlowchartElementCategory.INPUT_OUTPUT.value:
                s += f"{elem_id}.shape: parallelogram\n"
            case FlowchartElementCategory.SUBROUTINE.value:
                pass # nothing to do
            case _:
                raise ValueError(f"Unknown flowchart element category: {category}")

        return s

    @staticmethod
    def wrap_relation(category: str, label: str, target_id: int) -> str:
        match category:
            case FlowchartRelationCategory.ARROW.value:
                if label.strip() == "":
                    return f"->{target_id}\n"
                return f"->{target_id}: {label}\n"

            case FlowchartRelationCategory.OPEN_LINK.value:
                if label.strip() == "":
                    return f"--{target_id}\n"
                return f"--{target_id}: {label}\n"

            case FlowchartRelationCategory.DOTTED_ARROW.value:
                if label.strip() == "":
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


    def get_text(self, obj: Relation | Element) -> str:
        pass

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        assert isinstance(diagram_representation, FlowchartRepresentation)

        body: str = ""
        for identifier, element in enumerate(diagram_representation.elements):
            element_text = ""
            if len(element.inner_text) > 0:
                element_text += f"{' '.join(element.inner_text)}\\n"
            relation_text = ""
            if len(element.outer_text) > 0:
                relation_text += f"{' '.join(element.outer_text)}\\n"

            element_text = element_text.strip()

            body += self.wrap_element(element.category, element_text, identifier)

        body += "\n"
        for relation in diagram_representation.relations:
            if relation.source_id is None or relation.target_id is None:
                continue
            body += f"{relation.source_id}"

            relation_text = ""
            if len(relation.source_text) > 0:
                relation_text += f"{' '.join(relation.source_text)}\\n"
            if len(relation.middle_text) > 0:
                relation_text += f"{' '.join(relation.middle_text)}\\n"
            if len(relation.inner_text) > 0:
                relation_text += f"{' '.join(relation.inner_text)}\\n"
            if len(relation.target_text) > 0:
                relation_text += f"{' '.join(relation.target_text)}\\n"

            relation_text = relation_text.strip()

            body += self.wrap_relation(relation.category, relation_text, relation.target_id)

        outcome: TransducerOutcome = TransducerOutcome(diagram_id, WellKnownMarkupLanguage.D2_LANG.value, body)
        return outcome