from enum import IntEnum
from typing import Dict

from src.flowchart_element_category import FlowchartExtraElementCategory
from src.representation.flowchart_representation.element import FlowchartElementCategory
from src.representation.flowchart_representation.relation import FlowchartRelationCategory
from src.wellknown_diagram import WellKnownDiagram


class FlowchartElementCategoryIndex(IntEnum):
    STATE = 1
    FINAL_STATE = 2
    TEXT = 3
    ARROW = 4
    CONNECTION = 5
    DATA = 6
    DECISION = 7
    PROCESS = 8
    TERMINATOR = 9
    ARROW_HEAD = 10
    ARROW_TAIL = 11


class Lookup:
    table_target_int_to_flowchart_category_str: Dict[int, str] = {
        FlowchartElementCategoryIndex.PROCESS.value: FlowchartElementCategory.PROCESS.value,
         FlowchartElementCategoryIndex.DECISION.value: FlowchartElementCategory.DECISION.value,
         FlowchartElementCategoryIndex.TERMINATOR.value: FlowchartElementCategory.TERMINAL.value,
         FlowchartElementCategoryIndex.DATA.value: FlowchartElementCategory.INPUT_OUTPUT.value,
         FlowchartElementCategoryIndex.CONNECTION.value: FlowchartElementCategory.CIRCLE.value,
         FlowchartElementCategoryIndex.ARROW.value: FlowchartRelationCategory.ARROW.value,
         FlowchartElementCategoryIndex.TEXT.value: FlowchartExtraElementCategory.TEXT.value,
         FlowchartElementCategoryIndex.ARROW_HEAD.value: FlowchartExtraElementCategory.ARROW_HEAD.value,
         FlowchartElementCategoryIndex.ARROW_TAIL.value: FlowchartExtraElementCategory.ARROW_TAIL.value
    }

    table_target_int_to_graph_category_str: Dict[int, str] = {
        FlowchartElementCategoryIndex.ARROW.value: FlowchartRelationCategory.ARROW.value,
        FlowchartElementCategoryIndex.STATE.value: FlowchartElementCategory.CIRCLE.value,
        FlowchartElementCategoryIndex.FINAL_STATE.value: FlowchartElementCategory.CIRCLE.value,
        FlowchartElementCategoryIndex.TEXT.value: FlowchartExtraElementCategory.TEXT.value,
        FlowchartElementCategoryIndex.ARROW_HEAD.value: FlowchartExtraElementCategory.ARROW_HEAD.value,
        FlowchartElementCategoryIndex.ARROW_TAIL.value: FlowchartExtraElementCategory.ARROW_TAIL.value
    }


    table_target_int_to_str_by_diagram_id: Dict[str, Dict[int, str]] = {
        WellKnownDiagram.FLOW_CHART.value: table_target_int_to_flowchart_category_str,
        WellKnownDiagram.GRAPH_DIAGRAM.value: table_target_int_to_graph_category_str,
    }

    table_str_to_target_int_by_diagram_id: Dict[str, Dict[str, int]] = {
        WellKnownDiagram.GRAPH_DIAGRAM.value: dict((v, k) for k,v in table_target_int_to_graph_category_str.items()),
        WellKnownDiagram.FLOW_CHART.value: dict((v, k) for k, v in table_target_int_to_flowchart_category_str.items()),
    }
