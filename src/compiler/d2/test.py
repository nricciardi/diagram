import unittest

from src.compiler.d2.flowchart_compiler import FlowchartToD2Compiler

class TestFlowchartToD2Compiler(unittest.TestCase):

    def setUp(self):
        self.compiler = FlowchartToD2Compiler('test_compiler')

    def test_compile(self):
        payload = """
        A: Circle
        A.shape: circle
        B: Process
        B.style.border-radius: 8
        C: Decision
        C.shape: diamond
        D: Input Output
        D.shape: parallelogram
        E: Subroutine
        F: Terminal state
        F.shape: oval
        A->B: Arrow
        B->C: Dotted Arrow {
            style: {
                stroke-dash: 3
            }
        }
        C->D
        D->E
        E->F
        """
        output_path = "test_d2.png"
        markuplang_file_path = 'test_d2.d2'
        self.compiler.compile(payload=payload, output_path=output_path, markuplang_file_path=markuplang_file_path)


if __name__ == '__main__':
    unittest.main()
