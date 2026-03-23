from __future__ import annotations

from typing import Literal, TypeAlias

BanTeam: TypeAlias = Literal["blue", "red"]
BanTurn: TypeAlias = tuple[BanTeam, int, int]

BAN_SEQUENCE: tuple[BanTurn, ...] = (
    ("blue", 1, 1),
    ("red", 1, 2),
    ("blue", 2, 3),
    ("red", 2, 4),
    ("blue", 3, 5),
    ("red", 3, 6),
    ("red", 4, 13),
    ("blue", 4, 14),
    ("red", 5, 15),
    ("blue", 5, 16),
)
