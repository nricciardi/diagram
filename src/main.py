from core.orchestrator.orchestrator import Orchestrator
from src.transducer.d2l.flowchart_transducer import FlowchartToD2Transducer
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer


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
            # TODO
        ]
    )


if __name__ == '__main__':
    main()