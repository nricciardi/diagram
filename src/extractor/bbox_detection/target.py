from enum import IntEnum


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