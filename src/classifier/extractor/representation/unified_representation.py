from src.classifier.extractor.representation.representation import DiagramRepresentation
from dataclasses import dataclass


@dataclass
class UnifiedDiagramRepresentation(DiagramRepresentation):
    """
    Unified diagram representation for general purpose uses
    """


    # TODO: inserite qui i campi che vi servono (NO id del diagramma, ci pensa l'orchestratore)
    # TIP: probabilmente una sottoclasse per i nodi ecc non sarebbe male