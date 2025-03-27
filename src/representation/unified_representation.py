from src.representation.representation import DiagramRepresentation
from dataclasses import dataclass


@dataclass
class UnifiedDiagramRepresentation(DiagramRepresentation):
    """
    Unified diagram representation for general purpose uses
    """

    def dump(self, output_path: str):
        raise NotImplemented()      # TODO

    def load(self, input_path: str):
        raise NotImplemented()      # TODO

    # TODO: inserite qui i campi che vi servono (NO id del diagramma, ci pensa l'orchestratore)
    # TIP: probabilmente una sottoclasse per i nodi ecc non sarebbe male