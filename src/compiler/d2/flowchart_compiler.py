from typing import List

from core.compiler.compiler import Compiler
from src.wellknown_markuplang import WellKnownMarkupLanguage

import os
import subprocess

import logging

logger = logging.getLogger(__name__)

class FlowchartToD2Compiler(Compiler):
    def compile(self, payload: str, output_path: str, dump_markuplang_file: bool = True,
                markuplang_file_path: str | None = None):
        if markuplang_file_path is None:
            markuplang_file_path = 'test_d2.d2'

        with open(markuplang_file_path, 'w') as file:
            file.write(payload)

        try:
            subprocess.run(["d2", markuplang_file_path, output_path])
            # d2.render(markuplang_file_path, output_path, format="png")
        except subprocess.CalledProcessError as e:
            logging.debug(f"Subprocess failed with exit code {e.returncode}")
            logging.debug(f"Command: {e.cmd}")
            logging.debug(f"Output: {e.output}")
            logging.debug(f"Stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            logging.debug("The 'd2' CLI tool is not installed or not found in PATH.")
            raise

        if dump_markuplang_file is not True:
            os.remove(markuplang_file_path)

    def compatible_markup_languages(self) -> List[str]:
        return [WellKnownMarkupLanguage.MERMAID.value]
