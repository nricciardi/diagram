import unittest
from typing import List, Type

from core.classifier.classifier import Classifier
from core.compiler.compiler import Compiler
from core.extractor.extractor import Extractor
from core.image.image import Image
from core.orchestrator.orchestrator import Orchestrator
from core.representation.representation import DiagramRepresentation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer

FAKE_DIAGRAM_TYPE_1 = "fake-diagram-1"
FAKE_DIAGRAM_TYPE_2 = "fake-diagram-2"
FAKE_DIAGRAM_TYPE_3 = "fake-diagram-3"
FAKE_MARKUP_LANGUAGE_1 = "fake-markuplang-1"
FAKE_MARKUP_LANGUAGE_2 = "fake-markuplang-2"
FAKE_PAYLOAD_1 = "fake-payload-1"
FAKE_PAYLOAD_2 = "fake-payload-2"

class MockClassifier(Classifier):

    n_calls = -1
    mock_classifications = [FAKE_DIAGRAM_TYPE_1, FAKE_DIAGRAM_TYPE_2]

    def classify(self, image: Image) -> str:
        self.n_calls += 1
        return self.mock_classifications[self.n_calls % len(self.mock_classifications)]

class MockDiagramRepresentation1(DiagramRepresentation):

    def dump(self, output_path: str):
        pass

    def load(self, input_path: str):
        pass

class MockDiagramRepresentation2(DiagramRepresentation):

    def dump(self, output_path: str):
        pass

    def load(self, input_path: str):
        pass


class MockExtractor1(Extractor):

    def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:
        return MockDiagramRepresentation1()

    def compatible_diagrams(self) -> List[str]:
        return [
            FAKE_DIAGRAM_TYPE_1
        ]

class MockExtractor2(Extractor):

    def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:
        return MockDiagramRepresentation2()

    def compatible_diagrams(self) -> List[str]:
        return [
            FAKE_DIAGRAM_TYPE_2
        ]

class MockExtractor3(Extractor):

    def extract(self, diagram_id: str, image: Image) -> DiagramRepresentation:
        raise ValueError("this doesn't have to be called")

    def compatible_diagrams(self) -> List[str]:
        return [
            FAKE_DIAGRAM_TYPE_3
        ]


class MockTransducer1(Transducer):

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        return TransducerOutcome(diagram_id, FAKE_MARKUP_LANGUAGE_1, FAKE_PAYLOAD_1)

    def compatible_diagrams(self) -> List[str]:
        return [
            FAKE_DIAGRAM_TYPE_1
        ]

    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [
            MockDiagramRepresentation1
        ]

class MockTransducer2(Transducer):

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        return TransducerOutcome(diagram_id, FAKE_MARKUP_LANGUAGE_2, FAKE_PAYLOAD_2)

    def compatible_diagrams(self) -> List[str]:
        return [
            FAKE_DIAGRAM_TYPE_2
        ]

    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [
            MockDiagramRepresentation2
        ]

class MockCompiler1(Compiler):

    def compile(self, payload: str, output_path: str, **kwargs):
        assert payload, FAKE_PAYLOAD_1

    def compatible_markup_languages(self) -> List[str]:
        return [
            FAKE_MARKUP_LANGUAGE_1
        ]

class MockCompiler2(Compiler):

    def compile(self, payload: str, output_path: str, **kwargs):
        assert payload, FAKE_PAYLOAD_2

    def compatible_markup_languages(self) -> List[str]:
        return [
            FAKE_MARKUP_LANGUAGE_2
        ]

class MockImage(Image):
    pass


class TestOrchestrator(unittest.TestCase):

    @staticmethod
    def default_orchestrator() -> Orchestrator:
        orchestrator = Orchestrator(
            classifier=MockClassifier(),
            extractors=[
                MockExtractor1("mock-extractor-1"),
                MockExtractor2("mock-extractor-2"),
                MockExtractor3("mock-extractor-3"),     # this should not be used
            ],
            transducers=[
                MockTransducer1("mock-transducer-1"),
                MockTransducer2("mock-transducer-2"),
            ],
            compilers=[
                MockCompiler1("mock-compiler-1"),
                MockCompiler2("mock-compiler-2"),
            ]
        )

        return orchestrator

    def test_sequential_image2diagram(self):
        orchestrator = TestOrchestrator.default_orchestrator()

        input = MockImage()

        outcomes: List[TransducerOutcome] = []
        outcomes.extend(orchestrator.image2diagram(image=input))
        outcomes.extend(orchestrator.image2diagram(image=input))

        assert len(outcomes), 2

        assert outcomes[0].payload, FAKE_PAYLOAD_1
        assert outcomes[1].payload, FAKE_PAYLOAD_2


    def test_parallel_image2diagram(self):
        orchestrator = TestOrchestrator.default_orchestrator()

        input = MockImage()

        outcomes: List[TransducerOutcome] = []
        outcomes.extend(orchestrator.image2diagram(image=input, parallelization=True))
        outcomes.extend(orchestrator.image2diagram(image=input, parallelization=True))

        assert len(outcomes), 2

        assert outcomes[0].payload, FAKE_PAYLOAD_1
        assert outcomes[1].payload, FAKE_PAYLOAD_2


    def test_parallel_images2diagrams(self):

        n = 100
        inputs = [MockImage() for _ in range(n)]

        orchestrator = TestOrchestrator.default_orchestrator()

        outcomes: List[TransducerOutcome] = orchestrator.images2diagrams(inputs, parallelization=True)

        assert len(outcomes), n