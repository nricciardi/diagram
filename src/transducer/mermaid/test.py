import unittest
import sys, os

from src.wellknown_markuplang import WellKnownMarkupLanguage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.transducer.outcome import TransducerOutcome
from src.representation.flowchart_representation.element import FlowchartElementCategory
from src.representation.flowchart_representation.flowchart_representation import Element, Relation, \
    FlowchartRepresentation
from typing import Dict, List

from src.representation.flowchart_representation.relation import FlowchartRelationCategory
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer
from src.wellknown_diagram import WellKnownDiagram


class TestFlowchartToMermaidTransducer(unittest.TestCase):

    def setUp(self):
        self.transducer = FlowchartToMermaidTransducer("test_transducer")

    def test_compatible_diagrams(self):
        compatible_diagrams = self.transducer.compatible_diagrams()
        assert WellKnownDiagram.FLOW_CHART.value in compatible_diagrams, "Flowchart should be compatible"
        assert WellKnownDiagram.GRAPH_DIAGRAM.value in compatible_diagrams, "Graph diagram should be compatible"

    def test_transduce(self):
        id_a, id_b, id_c = "0", "1", "2"

        elements: list[Element] = [
            Element(FlowchartElementCategory.CIRCLE.value, ["Start_Node"], []),
            Element(FlowchartElementCategory.PROCESS.value, ["i++"], []),
            Element(FlowchartElementCategory.DECISION.value, ["if i > 5"], []),
        ]

        relations: List[Relation] = [
            Relation(FlowchartRelationCategory.ARROW.value, id_a, id_b, ["int i = 0"], [], [], []),
            Relation(FlowchartRelationCategory.DOTTED_ARROW.value, id_b, id_c, [""], [], [], []),
        ]

        representation = FlowchartRepresentation(elements, relations)
        expected_outcome = TransducerOutcome(diagram_id="test_diagram", markup_language=WellKnownMarkupLanguage.MERMAID.value,
                                             payload="""flowchart TD
	0((Start_Node))
	1(i++)
	2{if i > 5}

	0-->|int i = 0|1
	1-.->2
""")

        assert expected_outcome == self.transducer.transduce("test_diagram", representation), \
            "Elaborated outcome should match expected outcome"


if __name__ == "__main__":
    unittest.main()