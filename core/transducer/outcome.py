from dataclasses import dataclass

from core.utils.dumpable import DumpableMixin
from core.utils.loadable import LoadableMixin
from src.wellknown_diagram import WellKnownDiagram


@dataclass(frozen=True, slots=True)
class TransducerOutcome(DumpableMixin, LoadableMixin):
    diagram_id: str
    markup_language: str
    payload: str

    def dump(self, output_path: str):
        with open(output_path, "w") as file:
            file.write(self.payload)

    @staticmethod
    def load(input_path: str):
        raise NotImplemented()