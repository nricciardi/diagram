from typing import List

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



    def __par_image2diagram(self, outputs_path: str, image: Image):
        """
        Convert image to diagram in parallel
        """

        diagram_id = self.__classifier.classify(image)


    def __compatible_extractors(self, diagram_id: str) -> List[Extractor]:
        """
        Return reference of compatible extractors with diagram identifier
        """

        compatible_extractors = list(
            extractor for extractor in self.__extractors if diagram_id in extractor.compatible_diagrams()
        )

        return compatible_extractors

    def __compatible_transducers(self, diagram_id: str) -> List[Extractor]:
        """
        Return reference of compatible extractors with diagram identifier
        """

        compatible_extractors = list(
            extractor for extractor in self.__extractors if diagram_id in extractor.compatible_diagrams()
        )

        return compatible_extractors


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

    def __seq_trasduce(self, diagram_id: str, image: Image) -> List[Outcome]:
        """
        Trasduce representation sequentially
        """













