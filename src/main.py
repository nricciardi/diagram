from typing import List, Optional
import argparse
import os
import sys
import logging
from pathlib import Path
import torch
from core.classifier.classifier import Classifier
from core.extractor.extractor import Extractor
from core.image.image import Image
from core.image.tensor_image import TensorImage
from core.orchestrator.orchestrator import Orchestrator
from src.classifier.classifier import GNRClassifier
from src.extractor.bbox_detection import load_model
from src.extractor.flowchart.gnr_extractor import GNRFlowchartExtractor
from src.transducer.d2.flowchart_transducer import FlowchartToD2Transducer
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer
from src.compiler.d2.flowchart_compiler import FlowchartToD2Compiler
from src.compiler.mermaid.flowchart_compiler import FlowchartToMermaidCompiler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
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
        format="%(asctime)s [%(levelname)s] %(message)s",
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
        default=None,
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

    return parser.parse_args()


def main(classifier: Classifier, extractors: List[Extractor], inputs_paths: List[str], parallelization: bool, then_compile: bool, outputs_dir_path: Optional[str]):

    images: List[Image] = [TensorImage.from_str(path) for path in inputs_paths]

    orchestrator = Orchestrator(
        classifier=classifier,
        extractors=extractors,
        transducers=[
            FlowchartToMermaidTransducer("flowchart-to-mermaid-transducer"),
            FlowchartToD2Transducer("flowchart-to-d2-transducer"),
        ],
        compilers=[
            FlowchartToMermaidCompiler("flowchart-to-mermaid-compiler"),
            FlowchartToD2Compiler("flowchart-to-d2-compiler")
        ],
    )

    orchestrator.images2diagrams(
        images,
        parallelization=parallelization,
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


    classifier = GNRClassifier(model_path=args.classifier)
    extractors = [
        GNRFlowchartExtractor(
            bbox_detector=load_model(args.bbox_detector, torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            identifier="gnr-flowchart-extractor",
            bbox_trust_threshold=0.7,
            parallelization=args.parallelize
        )
    ]

    main(
        classifier=classifier,
        extractors=extractors,
        inputs_paths=args.input,
        parallelization=args.parallelize,
        then_compile=args.then_compile,
        outputs_dir_path=str(args.outputs_dir_path)
    )
