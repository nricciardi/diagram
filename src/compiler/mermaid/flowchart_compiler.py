import base64
import io, requests
import logging

import cv2
import numpy as np
from typing import List
from core.compiler.compiler import Compiler
from src.wellknown_markuplang import WellKnownMarkupLanguage


class FlowchartToMermaidCompiler(Compiler):
    def compatible_markup_languages(self) -> List[str]:
        return [WellKnownMarkupLanguage.MERMAID.value]

    def compile(self, payload: str, output_path: str, dump_markuplang_file: bool = False, markuplang_file_path: str | None = None):
        try:
            mm(payload, output_path)
        except Exception as e:
            logging.error(f"Mermaid can not be compiled due to error: {e}")


def mm(graph: str, output_path: str):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")

    url = 'https://mermaid.ink/img/' + base64_string
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error fetching image: {response.status_code}, {response.text}")
    if "image" not in response.headers.get("Content-Type", ""):
        raise Exception(f"Invalid content type: {response.headers.get('Content-Type')}\nResponse: {response.text}")

    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if not cv2.imwrite(f"{output_path}.png", img):
        raise Exception(f"Could not write image to {output_path}.png")
