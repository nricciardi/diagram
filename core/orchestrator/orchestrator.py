import hashlib
import os
import logging
from datetime import datetime
from typing import List, Dict
from collections import defaultdict
from core.classifier.classifier import Classifier
from core.compiler.compiler import Compiler
from core.extractor.extractor import Extractor
from core.image.image import Image
from core.representation.representation import DiagramRepresentation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer

logger = logging.getLogger(__name__)

class Orchestrator:

    def __init__(self, classifier: Classifier, extractors: List[Extractor], transducers: List[Transducer], compilers: List[Compiler]):
        self.__classifier = classifier
        self.__extractors = extractors
        self.__transducers = transducers
        self.__compilers = compilers

    @classmethod
    def build_output_path(cls, outputs_dir_path: str, diagram_id: str, markup_language: str, make_unique: bool = True, unique_strength: int = 8) -> str:

        file_name = f"{diagram_id}__{markup_language}"

        if make_unique:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            hash_value = hashlib.md5(now_str.encode()).hexdigest()
            unique = hash_value[:min(len(hash_value) - 1, unique_strength)]

            file_name += f"__{unique}"

        return os.path.join(outputs_dir_path, file_name)

    async def image2diagram(self, image: Image, parallelization: bool = False, then_compile: bool = True, outputs_dir_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert one handwritten image to digital diagram

        :param outputs_dir_path: directory in which outputs will be dumped
        :param image: input image
        :param parallelization: enable parallelization
        :param then_compile: True if compile outcomes
        :return: all transducer outcomes, based on extractors and transducers more than one outcome can be produced
        """

        if parallelization:
            return await self.__par_image2diagram(image, then_compile=then_compile, outputs_path=outputs_dir_path)
        else:
            return await self.__seq_image2diagram(image, then_compile=then_compile, outputs_dir_path=outputs_dir_path)

    async def __classify(self, image: Image) -> str:
        logger.info("classify image...")
        logger.debug(image)

        diagram_id = await self.__classifier.classify(image)

        logger.info(f"image was classified as {diagram_id}")

        return diagram_id

    async def __seq_image2diagram(self, image: Image, then_compile: bool, outputs_dir_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert image to diagram sequentially
        """

        diagram_id = await self.__classify(image)

        logger.info(f"extract image...")
        diagram_representations: List[DiagramRepresentation] = await self.__seq_extraction(diagram_id, image)

        logger.info(f"{len(diagram_representations)} diagram representation(s) found")
        logger.debug(diagram_representations)

        outcomes: List[TransducerOutcome] = []
        for diagram_representation in diagram_representations:
            logger.info(f"transduce {type(diagram_representation)} type...")
            o = await self.__seq_transduce(diagram_id, diagram_representation)

            logger.info(f"transduction done: {len(o)} outcomes")
            logger.debug(o)

            outcomes.extend(o)

        if not then_compile:
            return outcomes

        await self.__seq_compile_transducer_outcomes(outcomes, outputs_dir_path)

        return outcomes

    async def __par_image2diagram(self, image: Image, then_compile: bool, outputs_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert image to diagram in parallel
        """

        diagram_id = await self.__classify(image)

        raise NotImplemented()


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


    async def __seq_extraction(self, diagram_id: str, image: Image) -> List[DiagramRepresentation]:
        """
        Extract representations from image, using extractors sequentially
        """

        compatible_extractors = self.__compatible_extractors(diagram_id)

        diagram_representations: List[DiagramRepresentation] = []

        for extractor in compatible_extractors:
            logger.debug(f"extract using {extractor.identifier}")
            representation: DiagramRepresentation = await extractor.extract(diagram_id, image)
            diagram_representations.append(representation)

        return diagram_representations

    async def __seq_transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> List[TransducerOutcome]:
        """
        Transduce representation sequentially
        """

        compatible_transducer: List[Transducer] = self.__compatible_transducers(diagram_id, diagram_representation)

        outcomes: List[TransducerOutcome] = []
        for transducer in compatible_transducer:
            logger.debug(f"transduce using {transducer.identifier}")
            outcome = await transducer.transduce(diagram_id, diagram_representation)
            outcomes.append(outcome)

        return outcomes

    async def __seq_compile_transducer_outcomes(self, outcomes: List[TransducerOutcome], outputs_dir_path: str):

        outcomes_by_markuplang: Dict[str, List[TransducerOutcome]] = defaultdict(list)  # { markuplang: [outcomes] }
        for outcome in outcomes:
            outcomes_by_markuplang[outcome.markup_language].append(outcome)

        logger.info(f"outcomes are grouped into {len(outcomes_by_markuplang.keys())} groups: {outcomes_by_markuplang.keys()}")
        logger.debug(outcomes_by_markuplang)

        for markuplang, outcomes in outcomes_by_markuplang.items():
            for compiler in self.__compilers:
                if markuplang in compiler.compatible_markup_languages():
                    for outcome in outcomes:
                        logger.info(f"compile using {compiler.identifier}...")
                        logger.debug(compiler)
                        logger.debug(outcome)

                        await compiler.compile(
                            outcome.payload,
                            Orchestrator.build_output_path(
                                outputs_dir_path,
                                outcome.diagram_id,
                                outcome.markup_language
                            )
                        )








