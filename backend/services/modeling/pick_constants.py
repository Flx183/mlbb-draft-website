from __future__ import annotations

from typing import Literal, TypeAlias

PickTeam: TypeAlias = Literal["blue", "red"]
PickTurn: TypeAlias = tuple[PickTeam, int, int]

FIRST_PICK_PHASE_REQUIRED_BANS = 6
SECOND_PICK_PHASE_REQUIRED_BANS = 10
FIRST_PICK_PHASE_TURNS = 6

PICK_SEQUENCE: tuple[PickTurn, ...] = (
    ("blue", 1, 7),
    ("red", 1, 8),
    ("red", 2, 9),
    ("blue", 2, 10),
    ("blue", 3, 11),
    ("red", 3, 12),
    ("red", 4, 17),
    ("blue", 4, 18),
    ("blue", 5, 19),
    ("red", 5, 20),
)
