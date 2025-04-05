import unittest

from src.compiler.mermaid.flowchart_compiler import FlowchartToMermaidCompiler


class TestFlowchartToMermaidCompiler(unittest.TestCase):

    def setUp(self):
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


if __name__ == '__main__':
    unittest.main()
