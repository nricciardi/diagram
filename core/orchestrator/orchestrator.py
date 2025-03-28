from typing import List

from core.classifier.classifier import Classifier
from core.extractor import Extractor
from core.image import Image
from core.transducer.transducer import Transducer


class Orchestrator:

    def __init__(self, classifier: Classifier, extractors: List[Extractor], transducers: List[Transducer]):
        self.__classifier = classifier
        self.__extractors = extractors
        self.__transducers = transducers


    def draw2diagram(self, outputs_path: str, image: Image):
        """
        Convert handwritten image to digital diagram

        :param outputs_path: directory in which outputs will be dumped
        :param image: input image
        :return:
        """

        diagram_id = self.__classifier.classify(image)

        compatible_extractors = list(extractor for extractor in self.__extractors if diagram_id in extractor.compatible_diagrams())

