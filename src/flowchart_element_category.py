from enum import IntEnum, Enum
from typing import Dict

from src.extractor.bbox_detection.target import FlowchartElementCategoryIndex
from src.representation.flowchart_representation.element import FlowchartElementCategory
from src.representation.flowchart_representation.relation import FlowchartRelationCategory



class FlowchartExtraElementCategory(Enum):
    ARROW_HEAD = "head"
    ARROW_TAIL = "tail"
    TEXT = "text"
    


class Lookup:
    table: Dict[int, str] = {
        FlowchartElementCategoryIndex.PROCESS.value: FlowchartElementCategory.PROCESS.value,
         FlowchartElementCategoryIndex.DECISION.value: FlowchartElementCategory.DECISION.value,
         FlowchartElementCategoryIndex.TERMINATOR.value: FlowchartElementCategory.TERMINAL.value,
         FlowchartElementCategoryIndex.DATA.value: FlowchartElementCategory.INPUT_OUTPUT.value,
         FlowchartElementCategoryIndex.CONNECTION.value: FlowchartElementCategory.CIRCLE.value,
         FlowchartElementCategoryIndex.ARROW.value: FlowchartRelationCategory.ARROW.value,
         FlowchartElementCategoryIndex.STATE.value: FlowchartElementCategory.CIRCLE.value,
         FlowchartElementCategoryIndex.FINAL_STATE.value: FlowchartElementCategory.CIRCLE.value,
         FlowchartElementCategoryIndex.TEXT.value: FlowchartExtraElementCategory.TEXT.value,
         FlowchartElementCategoryIndex.ARROW_HEAD.value: FlowchartExtraElementCategory.ARROW_HEAD.value,
         FlowchartElementCategoryIndex.ARROW_TAIL.value: FlowchartExtraElementCategory.ARROW_TAIL.value
    }
