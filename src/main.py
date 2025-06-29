from typing import List, Optional
import argparse
import os
import sys
import logging
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.compiler.compiler import Compiler
from core.transducer.transducer import Transducer
from src.extractor.bbox_detection.target import FlowchartElementCategoryIndex
from core.classifier.classifier import Classifier
from core.extractor.extractor import Extractor
from core.image.image import Image
from core.image.tensor_image import TensorImage
from core.orchestrator.orchestrator import Orchestrator
from src import DEVICE
from src.classifier.classifier import GNRClassifier
from src.extractor.bbox_detection import load_model
from src.extractor.flowchart.gnr_extractor import GNRFlowchartExtractor
from src.extractor.text_extraction.text_extractor import TrOCRTextExtractorSmall, TrOCRTextExtractorBaseHandwritten
from src.transducer.d2.flowchart_transducer import FlowchartToD2Transducer
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer
from src.compiler.d2.flowchart_compiler import FlowchartToD2Compiler
from src.compiler.mermaid.flowchart_compiler import FlowchartToMermaidCompiler
from src.wellknown_markuplang import WellKnownMarkupLanguage

logger = logging.getLogger(__name__)

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def configure_logger(level_name: str):
    level = LOG_LEVELS.get(level_name.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(name)s] %(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)
    logger.debug(f"Logger configured with level: {level_name.upper()}")

    return logger


def validate_and_create_output_dir(path, logger):
    output_path = Path(path)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        logger.error(f"The path '{output_path}' exists but is not a directory.")
        sys.exit(1)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Process input files with optional parallelization and compilation.")

    # Optional flags
    parser.add_argument(
        "-p", "--parallelize",
        action="store_true",
        help="Enable parallel execution (optional, default: False)"
    )

    parser.add_argument(
        "-c", "--then-compile",
        action="store_true",
        help="Compile after processing (optional, default: False)"
    )

    parser.add_argument(
        "--outputs-dir-path",
        type=str,
        required=True,
        help="Path to the output directory for compiled results (required if --then-compile is enabled)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=LOG_LEVELS.keys(),
        help="Set the logging level (default: info)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Set device"
    )

    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="(GNR)Classifier weights"
    )

    parser.add_argument(
        "--bbox-detector",
        type=str,
        required=True,
        help="Bounding box detector weights"
    )

    parser.add_argument(
        "--input",
        required=True,
        nargs='+',
        type=str,
        help="List of input files from the local file system (required)"
    )

    # === THRESHOLDs ===
    parser.add_argument(
        "--element_precedent_over_arrow_in_text_association",
        action="store_true",
        help="element_precedent_over_arrow_in_text_association"
    )

    parser.add_argument(
        "--element_text_overlap_threshold",
        type=float,
        default=0.5,
        help="element_text_overlap_threshold"
    )

    parser.add_argument(
        "--element_text_distance_threshold",
        type=float,
        default=150,
        help="element_text_distance_threshold"
    )

    parser.add_argument(
        "--arrow_text_discard_distance_threshold",
        type=float,
        default=150.0,
        help="arrow_text_discard_distance_threshold"
    )

    parser.add_argument(
        "--arrow_text_inner_distance_threshold",
        type=float,
        default=2.0,
        help="arrow_text_inner_distance_threshold"
    )

    parser.add_argument(
        "--arrow_crop_delta_size_x",
        type=float,
        default=40.0,
        help="arrow_crop_delta_size_x"
    )

    parser.add_argument(
        "--arrow_crop_delta_size_y",
        type=float,
        default=25.0,
        help="arrow_crop_delta_size_y"
    )

    parser.add_argument(
        "--element_arrow_distance_threshold",
        type=float,
        default=150,
        help="element_arrow_distance_threshold"
    )

    return parser.parse_args()


def main(device: str, classifier: Classifier, extractors: List[Extractor], transducers: List[Transducer], compilers: List[Compiler], inputs_paths: List[str], parallelization: bool, then_compile: bool, outputs_dir_path: Optional[str]):
    images: List[Image] = []
    for path in inputs_paths:
        image = TensorImage.from_str(path)
        image.to_device(DEVICE)
        images.append(image)

    orchestrator = Orchestrator(
        classifier=classifier,
        extractors=extractors,
        transducers=transducers,
        compilers=compilers,
        extensions_lookup={
            WellKnownMarkupLanguage.D2_LANG.value: "d2",
            WellKnownMarkupLanguage.MERMAID.value: "mmd",
        }
    )

    orchestrator.to_device(device)

    orchestrator.images2diagrams(
        images,
        parallelization=parallelization,
        dump_markup=True,       # CLI always dumps markup files
        then_compile=then_compile,
        outputs_dir_path=outputs_dir_path
    )


if __name__ == '__main__':
    args = parse_args()
    logger = configure_logger(args.log_level)

    # Validate compile output directory
    if args.then_compile:
        if not args.outputs_dir_path:
            logger.error("'--outputs-dir-path' is required when '--then-compile' is enabled.")
            sys.exit(1)
        args.outputs_dir_path = validate_and_create_output_dir(args.outputs_dir_path, logger)

    # Validate input files
    for file in args.input:
        if not os.path.isfile(file):
            logger.error(f"Input file '{file}' does not exist.")
            sys.exit(1)

    # Show configuration
    logger.info("Selected options:")
    logger.info(f"- Parallelize: {args.parallelize}")
    logger.info(f"- Then compile: {args.then_compile}")
    if args.then_compile:
        logger.info(f"- Output directory: {args.outputs_dir_path}")
    logger.info(f"- Input files: {args.input}")


    classifier = GNRClassifier("gnr-classifier", model_path=args.classifier)
    text_digitizer = TrOCRTextExtractorSmall()
    # less expensive version, but with lower performance: TrOCRTextExtractorSmall()

    extractors = [
        GNRFlowchartExtractor(
            text_digitizer=text_digitizer,
            bbox_detector=load_model(args.bbox_detector, torch.device(DEVICE)),
            identifier="gnr-flowchart-extractor",
            bbox_trust_thresholds={
                FlowchartElementCategoryIndex.TEXT.value: 0.5,
                FlowchartElementCategoryIndex.ARROW.value: 0.5,
                FlowchartElementCategoryIndex.FINAL_STATE.value: 0.85,
                FlowchartElementCategoryIndex.STATE.value: 0.85,
                FlowchartElementCategoryIndex.ARROW_TAIL.value: 0.5,
                FlowchartElementCategoryIndex.ARROW_HEAD.value: 0.5,
                FlowchartElementCategoryIndex.CONNECTION.value: 0.5,
                FlowchartElementCategoryIndex.DATA.value: 0.5,
                FlowchartElementCategoryIndex.PROCESS.value: 0.5,
                FlowchartElementCategoryIndex.TERMINATOR.value: 0.5,
                FlowchartElementCategoryIndex.DECISION.value: 0.5,
            },
            parallelization=args.parallelize,
            element_precedent_over_arrow_in_text_association=args.element_precedent_over_arrow_in_text_association,
            element_text_overlap_threshold=args.element_text_overlap_threshold,
            element_text_distance_threshold=args.element_text_distance_threshold,
            arrow_text_discard_distance_threshold=args.arrow_text_discard_distance_threshold,
            arrow_text_inner_distance_threshold=args.arrow_text_inner_distance_threshold,
            arrow_crop_delta_size_x=args.arrow_crop_delta_size_x,
            arrow_crop_delta_size_y=args.arrow_crop_delta_size_y,
            element_arrow_distance_threshold=args.element_arrow_distance_threshold
        )
    ]

    main(
        device=args.device,
        classifier=classifier,
        extractors=extractors,
        inputs_paths=args.input,
        parallelization=args.parallelize,
        then_compile=args.then_compile,
        outputs_dir_path=str(args.outputs_dir_path),
        transducers=[
            # TODO: FlowchartToMermaidTransducer("flowchart-to-mermaid-transducer"),
            FlowchartToD2Transducer("flowchart-to-d2-transducer"),
        ],
        compilers=[
            FlowchartToMermaidCompiler("flowchart-to-mermaid-compiler"),
            FlowchartToD2Compiler("flowchart-to-d2-compiler")
        ]
    )
