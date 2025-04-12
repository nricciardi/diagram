import hashlib
import os
import logging
from dataclasses import dataclass
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
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


@dataclass
class Orchestrator:
    classifier: Classifier
    extractors: List[Extractor]
    transducers: List[Transducer]
    compilers: List[Compiler]

    @classmethod
    def _build_output_path(cls, outputs_dir_path: str, diagram_id: str, markup_language: str, make_unique: bool = True, unique_strength: int = 8) -> str:

        file_name = f"{diagram_id}__{markup_language}"

        if make_unique:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            hash_value = hashlib.md5(now_str.encode()).hexdigest()
            unique = hash_value[:min(len(hash_value) - 1, unique_strength)]

            file_name += f"__{unique}"

        return os.path.join(outputs_dir_path, file_name)

    @staticmethod
    def _arrange_outcomes_by_markuplang(outcomes: List[TransducerOutcome]):
        outcomes_by_markuplang: Dict[str, List[TransducerOutcome]] = defaultdict(list)  # { markuplang: [outcomes] }
        for outcome in outcomes:
            outcomes_by_markuplang[outcome.markup_language].append(outcome)

        logger.debug(outcomes_by_markuplang)

        return outcomes_by_markuplang

    def __classify(self, image: Image) -> str:
        logger.info("classify image...")
        logger.debug(image)

        diagram_id = self.__classifier.classify(image)

        logger.info(f"image was classified as {diagram_id}")

        return diagram_id

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


    def images2diagrams(self, images: List[Image], parallelization: bool = False, then_compile: bool = True, outputs_dir_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert a bulk of images

        :param outputs_dir_path: directory in which outputs will be dumped
        :param images: input images
        :param parallelization: enable parallelization
        :param then_compile: True if compile outcomes
        :return: all transducer outcomes, based on extractors and transducers more than one outcome can be produced
        """

        outcomes: List[TransducerOutcome] = []
        if parallelization:

            with ProcessPoolExecutor() as executor:
                tasks: List[Future] = []
                for index, image in enumerate(images):
                    logger.info(f"elaborate image n. {index}")

                    tasks.append(
                        executor.submit(self.image2diagram, image, parallelization, then_compile, outputs_dir_path)
                    )

                outcomes: List[TransducerOutcome] = []
                for task in tasks:
                    outcomes.extend(task.result())

                return outcomes

        else:
            for index, image in enumerate(images):
                logger.info(f"elaborate image n. {index}")
                outcomes.extend(self.image2diagram(image, parallelization, then_compile=then_compile, outputs_dir_path=outputs_dir_path))


        return outcomes

    def image2diagram(self, image: Image, parallelization: bool = False, then_compile: bool = True, outputs_dir_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert one handwritten image to digital diagram

        :param outputs_dir_path: directory in which outputs will be dumped
        :param image: input image
        :param parallelization: enable parallelization
        :param then_compile: True if compile outcomes
        :return: all transducer outcomes, based on extractors and transducers more than one outcome can be produced
        """

        if parallelization:
            return self.__par_image2diagram(image, then_compile=then_compile, outputs_dir_path=outputs_dir_path)
        else:
            return self.__seq_image2diagram(image, then_compile=then_compile, outputs_dir_path=outputs_dir_path)

    # ====> SEQ <====

    def __seq_image2diagram(self, image: Image, then_compile: bool, outputs_dir_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert image to diagram sequentially
        """

        diagram_id = self.__classify(image)

        logger.info(f"extract image...")
        diagram_representations: List[DiagramRepresentation] = self.__seq_extraction(diagram_id, image)

        logger.info(f"{len(diagram_representations)} diagram representation(s) found")
        logger.debug(diagram_representations)

        outcomes: List[TransducerOutcome] = []
        for diagram_representation in diagram_representations:
            logger.info(f"transduce {type(diagram_representation)} type...")
            o = self.__seq_transduce(diagram_id, diagram_representation)

            logger.info(f"transduction done: {len(o)} outcomes")
            logger.debug(o)

            outcomes.extend(o)

        if not then_compile:
            return outcomes

        self.__seq_compile_transducer_outcomes(outcomes, outputs_dir_path)

        return outcomes


    def __seq_extraction(self, diagram_id: str, image: Image) -> List[DiagramRepresentation]:
        """
        Extract representations from image, using extractors sequentially
        """

        compatible_extractors = self.__compatible_extractors(diagram_id)

        diagram_representations: List[DiagramRepresentation] = []

        for extractor in compatible_extractors:
            logger.debug(f"extract using {extractor.identifier}")
            representation: DiagramRepresentation = extractor.extract(diagram_id, image)
            diagram_representations.append(representation)

        return diagram_representations

    def __seq_transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> List[TransducerOutcome]:
        """
        Transduce representation sequentially
        """

        compatible_transducer: List[Transducer] = self.__compatible_transducers(diagram_id, diagram_representation)

        outcomes: List[TransducerOutcome] = []
        for transducer in compatible_transducer:
            logger.debug(f"transduce using {transducer.identifier}")
            outcome = transducer.transduce(diagram_id, diagram_representation)
            outcomes.append(outcome)

        return outcomes

    def __seq_compile_transducer_outcomes(self, outcomes: List[TransducerOutcome], outputs_dir_path: str):

        outcomes_by_markuplang: Dict[str, List[TransducerOutcome]] = Orchestrator._arrange_outcomes_by_markuplang(outcomes)
        logger.info(f"outcomes are grouped into {len(outcomes_by_markuplang.keys())} groups: {outcomes_by_markuplang.keys()}")

        for markuplang, outcomes in outcomes_by_markuplang.items():
            for compiler in self.__compilers:
                if markuplang in compiler.compatible_markup_languages():
                    for outcome in outcomes:
                        logger.info(f"compile using {compiler.identifier}...")
                        logger.debug(compiler)
                        logger.debug(outcome)

                        compiler.compile(
                            outcome.payload,
                            Orchestrator._build_output_path(
                                outputs_dir_path,
                                outcome.diagram_id,
                                outcome.markup_language
                            )
                        )

    # ===> PAR <===

    def __par_image2diagram(self, image: Image, then_compile: bool, outputs_dir_path: str | None = None) -> List[TransducerOutcome]:
        """
        Convert image to diagram in parallel
        """

        diagram_id = self.__classify(image)

        outcomes = self.__par_elaboration(diagram_id, image)

        if not then_compile:
            return outcomes

        self.__par_compile(outcomes, outputs_dir_path)

        return outcomes


    def __par_elaboration(self, diagram_id: str, image: Image) -> List[TransducerOutcome]:
        """
        Elaborate image, using extractors and transducer in parallel
        """

        compatible_extractors = self.__compatible_extractors(diagram_id)

        with ProcessPoolExecutor() as executor:
            tasks: List[Future] = []
            for extractor in compatible_extractors:
                logger.debug(f"extract using {extractor.identifier}")

                tasks.append(
                    executor.submit(self._par_extract_using_extractor_and_transduce, extractor, diagram_id, image)
                )

            outcomes: List[TransducerOutcome] = []
            for task in tasks:
                outcomes.extend(task.result())

            return outcomes

    def _par_extract_using_extractor_and_transduce(self, extractor: Extractor, diagram_id: str, image: Image) -> List[TransducerOutcome]:
        diagram_representation: DiagramRepresentation = extractor.extract(diagram_id, image)

        compatible_transducers = self.__compatible_transducers(diagram_id, diagram_representation)

        with ProcessPoolExecutor() as executor:
            tasks: List[Future] = []
            for transducer in compatible_transducers:
                logger.info(f"transduce {type(diagram_representation)} type...")
                tasks.append(
                    executor.submit(transducer.transduce, diagram_id, diagram_representation)
                )

            outcomes: List[TransducerOutcome] = []
            for task in tasks:
                outcomes.append(task.result())

            return outcomes

    def __par_compile(self, outcomes: List[TransducerOutcome], outputs_dir_path: str):

        outcomes_by_markuplang: Dict[str, List[TransducerOutcome]] = Orchestrator._arrange_outcomes_by_markuplang(outcomes)
        logger.info(f"outcomes are grouped into {len(outcomes_by_markuplang.keys())} groups: {outcomes_by_markuplang.keys()}")

        def compile_using_compiler(compiler: Compiler, input: Dict[str, List[TransducerOutcome]]):
            for markuplang, outcomes in outcomes_by_markuplang:
                logger.info(f"compile {len(outcomes)} outcomes of markuplang: {markuplang}...")

                for markuplang, outcomes in input.items():
                    if markuplang in compiler.compatible_markup_languages():
                        for outcome in outcomes:
                            logger.info(f"compile using {compiler.identifier}...")
                            logger.debug(compiler)
                            logger.debug(outcome)

                            compiler.compile(
                                outcome.payload,
                                Orchestrator._build_output_path(
                                    outputs_dir_path,
                                    outcome.diagram_id,
                                    outcome.markup_language
                                )
                            )

        with ProcessPoolExecutor() as executor:
            tasks: List[Future] = []
            for compiler in self.__compilers:
                tasks.append(
                    executor.submit(compile_using_compiler, compiler, outcomes_by_markuplang)
                )

            _ = (task.result() for task in tasks)
