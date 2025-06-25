from typing import List

from core.compiler.compiler import Compiler
from src.wellknown_markuplang import WellKnownMarkupLanguage

import os
import subprocess

import logging

logger = logging.getLogger(__name__)
TMP_FILE = 'tmp.d2'


class FlowchartToD2Compiler(Compiler):
    def compile(self, payload: str, output_path: str):

        base_dir = os.path.dirname(output_path)
        tmp_file_path = os.path.join(base_dir, TMP_FILE)

        with open(tmp_file_path, 'w') as file:
            file.write(payload)

        try:
            subprocess.run(["d2", TMP_FILE, f"{output_path}.png"])
            # d2.render(markuplang_file_path, output_path, format="png")

        except subprocess.CalledProcessError as e:

            logging.debug(f"Subprocess failed with exit code {e.returncode}")
            logging.debug(f"Command: {e.cmd}")
            logging.debug(f"Output: {e.output}")
            logging.debug(f"Stderr: {e.stderr}")
            raise e

        except FileNotFoundError:
            logging.debug("The 'd2' CLI tool is not installed or not found in PATH.")
            raise

        finally:
            os.remove(tmp_file_path)


    def compatible_markup_languages(self) -> List[str]:
        return [ WellKnownMarkupLanguage.D2_LANG.value ]
