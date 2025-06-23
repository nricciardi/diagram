from typing import List

from core.compiler.compiler import Compiler
from src.wellknown_markuplang import WellKnownMarkupLanguage

import base64
import io, requests
from PIL import Image as im
import matplotlib.pyplot as plt


class FlowchartToMermaidCompiler(Compiler):
    def compatible_markup_languages(self) -> List[str]:
        return [WellKnownMarkupLanguage.MERMAID.value]

    def compile(self, payload: str, output_path: str, dump_markuplang_file: bool = False,
                markuplang_file_path: str | None = None):
        if dump_markuplang_file is True:
            with open(markuplang_file_path, 'w') as file:
                file.write(payload)

        mm(payload, output_path)


def mm(graph: str, output_path: str):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")

    url = 'https://mermaid.ink/img/' + base64_string
    response = requests.get(url)

    # Check if the response is actually an image
    if response.status_code != 200:
        raise Exception(f"Error fetching image: {response.status_code}, {response.text}")
    if "image" not in response.headers.get("Content-Type", ""):
        raise Exception(f"Invalid content type: {response.headers.get('Content-Type')}\nResponse: {response.text}")

    img = im.open(io.BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')  # allow to hide axis
    plt.savefig(output_path, dpi=1200)
