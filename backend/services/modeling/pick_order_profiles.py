from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PickOrderProfile:
    id: str
    title: str
    summary: str
    base_score_weight: float
    secure_power_weight: float
    flexibility_weight: float
    synergy_weight: float
    counter_weight: float
    role_completion_weight: float
    redundancy_penalty_weight: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


OPENER_PROFILE = PickOrderProfile(
    id="opener",
    title="Opener",
    summary="Prioritize secure power and flexibility while the board is still mostly hidden.",
    base_score_weight=0.72,
    secure_power_weight=0.18,
    flexibility_weight=0.10,
    synergy_weight=0.04,
    counter_weight=0.02,
    role_completion_weight=0.02,
    redundancy_penalty_weight=0.04,
)

BRIDGE_PROFILE = PickOrderProfile(
    id="bridge",
    title="Bridge",
    summary="Balance raw strength with ally synergy and clearer responses to revealed enemy picks.",
    base_score_weight=0.60,
    secure_power_weight=0.08,
    flexibility_weight=0.04,
    synergy_weight=0.12,
    counter_weight=0.09,
    role_completion_weight=0.05,
    redundancy_penalty_weight=0.05,
)

SETUP_PROFILE = PickOrderProfile(
    id="setup",
    title="Second-Phase Setup",
    summary="Use the second pick phase to shape the endgame with stronger counter and role-fit priorities.",
    base_score_weight=0.55,
    secure_power_weight=0.04,
    flexibility_weight=0.03,
    synergy_weight=0.08,
    counter_weight=0.12,
    role_completion_weight=0.12,
    redundancy_penalty_weight=0.06,
)

CLOSER_PROFILE = PickOrderProfile(
    id="closer",
    title="Closer",
    summary="Late picks should solve remaining role gaps and punish the enemy's nearly finished draft.",
    base_score_weight=0.50,
    secure_power_weight=0.02,
    flexibility_weight=0.02,
    synergy_weight=0.06,
    counter_weight=0.16,
    role_completion_weight=0.18,
    redundancy_penalty_weight=0.08,
)


def resolve_pick_order_profile(
    global_pick_index: int,
    phase_index: int,
    pick_order: int,
) -> PickOrderProfile:
    if global_pick_index <= 2:
        return OPENER_PROFILE
    if phase_index == 1 or pick_order <= 3:
        return BRIDGE_PROFILE
    if global_pick_index <= 8:
        return SETUP_PROFILE
    return CLOSER_PROFILE
