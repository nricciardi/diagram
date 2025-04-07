from core.orchestrator.orchestrator import Orchestrator
from src.transducer.d2.flowchart_transducer import FlowchartToD2Transducer
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer
from src.compiler.d2.flowchart_compiler import FlowchartToD2Compiler
from src.compiler.mermaid.flowchart_compiler import FlowchartToMermaidCompiler

def main():
    orchestrator = Orchestrator(
        classifier=None,        # TODO
        extractors=[
            # TODO
        ],
        transducers=[
            FlowchartToMermaidTransducer("flowchart-to-mermaid-transducer"),
            FlowchartToD2Transducer("flowchart-to-d2-transducer"),
        ],
        compilers=[
            FlowchartToMermaidCompiler("flowchart-to-mermaid-compiler"),
            FlowchartToD2Compiler("flowchart-to-d2-compiler")
        ]
    )


if __name__ == '__main__':
    main()