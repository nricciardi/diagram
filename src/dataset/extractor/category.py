from enum import IntEnum


class Category(IntEnum):
    STATE = 1
    FINAL_STATE = 2
    TEXT = 3
    ARROW = 4
    CONNECTION = 5
    DATA = 6
    DECISION = 7
    PROCESS = 8
    TERMINATOR = 9

    # TODO: head/tail
