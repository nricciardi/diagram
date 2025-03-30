from typing import List, Type

from core.classifier.classifier import Classifier
from core.extractor.extractor import Extractor
from core.image import Image
from core.representation.representation import DiagramRepresentation
from core.transducer.outcome import Outcome
from core.transducer.transducer import Transducer


class Orchestrator:

    def __init__(self, classifier: Classifier, extractors: List[Extractor], transducers: List[Transducer]):
        self.__classifier = classifier
        self.__extractors = extractors
        self.__transducers = transducers

    def image2diagram(self, outputs_path: str, image: Image, parallelization: bool = False):
        """
        Convert one handwritten image to digital diagram

        :param outputs_path: directory in which outputs will be dumped
        :param image: input image
        :param parallelization: enable parallelization
        :return:
        """

        if parallelization:
            self.__par_image2diagram(outputs_path, image)
        else:
            self.__seq_image2diagram(outputs_path, image)

    def __seq_image2diagram(self, outputs_path: str, image: Image):
        """
        Convert image to diagram sequentially
        """

        diagram_id = self.__classifier.classify(image)

        diagram_representations: List[DiagramRepresentation] = self.__seq_extraction(diagram_id, image)

        outcomes: List[Outcome] = self.__seq_transduce(diagram_id, )


    def __par_image2diagram(self, outputs_path: str, image: Image):
        """
        Convert image to diagram in parallel
        """

        diagram_id = self.__classifier.classify(image)


    def __compatible_extractors(self, diagram_id: str) -> List[Extractor]:
        """
        Return reference of compatible extractors given diagram identifier
        """

        compatible_extractors = list(
            extractor for extractor in self.__extractors if diagram_id in extractor.compatible_diagrams()
        )

        return compatible_extractors

    def __compatible_transducers(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> List[Transducer]:
        """
        Return reference of compatible transducer given diagram identifier and diagram representation type
        """

        compatible_transducer = []

        for transducer in self.__transducers:
            if diagram_id in transducer.compatible_diagrams():
                for compatible_representation_type in transducer.compatible_representations():
                    if isinstance(diagram_representation, compatible_representation_type):
                        compatible_transducer.append(transducer)

        return compatible_transducer


    def __seq_extraction(self, diagram_id: str, image: Image) -> List[DiagramRepresentation]:
        """
        Extract representations from image, using extractors sequentially
        """

        compatible_extractors = self.__compatible_extractors(diagram_id)

        diagram_representations: List[DiagramRepresentation] = []

        for extractor in compatible_extractors:
            representation: DiagramRepresentation = extractor.extract(diagram_id, image)
            diagram_representations.append(representation)

        return diagram_representations

    def __seq_transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> List[Outcome]:
        """
        Transduce representation sequentially
        """

        compatible_transducer: List[Transducer] = self.__compatible_transducers(diagram_id, diagram_representation)

        outcomes: List[Outcome] = []
        for transducer in compatible_transducer:
            outcome = transducer.transduce(diagram_id, diagram_representation)
            outcomes.append(outcome)

        return outcomes












