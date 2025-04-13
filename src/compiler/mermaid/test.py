import unittest

from core.representation.representation import DiagramRepresentation
from src.compiler.mermaid.flowchart_compiler import FlowchartToMermaidCompiler
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation, Element, Relation
from src.representation.flowchart_representation.element import FlowchartElementCategory
from src.representation.flowchart_representation.relation import FlowchartRelationCategory
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer


class TestFlowchartToMermaidCompiler(unittest.TestCase):

    def setUp(self):
        self.transducer = FlowchartToMermaidTransducer('test_transducer')
        self.compiler = FlowchartToMermaidCompiler('test_compiler')

    def test_compile(self):
        payload = """
graph LR;
    A--> B & C & D
    B--> A & E
    C--> A & E
    D--> A & E
    E--> B & C & D
"""

        payload2 = """---
title: test_diagram
---
graph 
a(("Start_Node"))
b["i++"]
c{"if i > 5")
a --->|int i = 0| b
b -.-> c"""
        # PAYLOAD2 IS THE ONE GIVING PROBLEMS
        output_path = 'test_mermaid.png'
        output_path2 = 'test_mermaid2.png'
        markup_lang_file_path = 'test_mermaid.mmd'
        markup_lang_file_path2 = 'test_mermaid.mmd'
        self.compiler.compile(payload=payload, output_path=output_path, markuplang_file_path=markup_lang_file_path)
        #self.compiler.compile(payload=payload2, output_path=output_path2, markuplang_file_path=markup_lang_file_path2)

    def test_transducer_compiler(self):
        diagram = FlowchartRepresentation(
            elements=[Element(FlowchartElementCategory.CIRCLE.value, ["Ciao"], []), Element(FlowchartElementCategory.CIRCLE.value, ["Ciao2"], [])],
            relations=[Relation(FlowchartRelationCategory.ARROW.value, "0", "1", [], [], [], [])]
        )
        output_path = 'test_mermaid3.png'
        outcome = self.transducer.transduce("diagram", diagram)
        markup_lang_file_path = 'test_mermaid.mmd'
        self.compiler.compile(payload=outcome.payload, output_path=output_path, dump_markuplang_file=False)


if __name__ == '__main__':
    unittest.main()
