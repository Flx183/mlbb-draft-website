from enum import Enum

class Role(Enum):
    EXP = "EXP"
    JUNGLE = "Jungle"
    MID = "Mid"
    GOLD = "Gold"
    ROAM = "Roam"

SLOT_TO_ROLE = {
    1: Role.EXP,
    2: Role.JUNGLE,
    3: Role.MID,
    4: Role.GOLD,
    5: Role.ROAM,
}