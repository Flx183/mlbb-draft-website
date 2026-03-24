from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

from backend.services.common.file_utils import load_json
from backend.services.modeling.pick_signal_model import ALL_SIGNAL_COLUMNS, POSITIVE_SIGNAL_COLUMNS

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "backend/data/modeling/models"
PICK_ORDER_PROFILE_PATH = MODEL_DIR / "pick_order_profiles.json"

BLEND_FEATURE_COLUMNS: tuple[str, ...] = ("prior_score", *ALL_SIGNAL_COLUMNS)
ORDER_LABELS = {
    1: ("opener", "Opener"),
    2: ("bridge", "Bridge"),
    3: ("bridge", "Bridge"),
    4: ("setup", "Second-Phase Setup"),
    5: ("closer", "Closer"),
}


class PickOrderProfilePayload(TypedDict):
    slot_index: int
    id: str
    title: str
    summary: str
    intercept: float
    base_score_weight: float
    secure_power_weight: float
    flexibility_weight: float
    synergy_weight: float
    counter_weight: float
    role_completion_weight: float
    redundancy_penalty_weight: float
    regularization_c: float


class PickOrderProfilesArtifact(TypedDict):
    source: str
    blend_features: list[str]
    slot_profiles: dict[str, PickOrderProfilePayload]


@dataclass(frozen=True)
class PickOrderProfile:
    slot_index: int
    id: str
    title: str
    summary: str
    intercept: float
    base_score_weight: float
    secure_power_weight: float
    flexibility_weight: float
    synergy_weight: float
    counter_weight: float
    role_completion_weight: float
    redundancy_penalty_weight: float
    regularization_c: float

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def _slot_identity(slot_index: int) -> tuple[str, str]:
    return ORDER_LABELS.get(slot_index, (f"slot-{slot_index}", f"Pick Slot {slot_index}"))


def _summary_from_coefficients(coefficients: dict[str, float], slot_index: int) -> str:
    labels = {
        "base_score_weight": "base ranker score",
        "secure_power_weight": "secure power",
        "flexibility_weight": "flexibility",
        "synergy_weight": "ally synergy",
        "counter_weight": "enemy counterplay",
        "role_completion_weight": "role completion",
        "redundancy_penalty_weight": "redundancy control",
    }
    ordered = [
        labels[name]
        for name, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)
        if abs(float(value)) > 1e-8
    ]
    if not ordered:
        return f"Learned blend profile for pick slot {slot_index}."
    return f"Learned blend emphasizing {', '.join(ordered[:3])} for pick slot {slot_index}."


def _row_to_profile(slot_index: int, intercept: float, coefficients: dict[str, float], regularization_c: float) -> PickOrderProfile:
    profile_id, profile_title = _slot_identity(slot_index)
    return PickOrderProfile(
        slot_index=slot_index,
        id=profile_id,
        title=profile_title,
        summary=_summary_from_coefficients(coefficients, slot_index),
        intercept=float(intercept),
        base_score_weight=float(coefficients.get("base_score_weight", 0.0)),
        secure_power_weight=float(coefficients.get("secure_power_weight", 0.0)),
        flexibility_weight=float(coefficients.get("flexibility_weight", 0.0)),
        synergy_weight=float(coefficients.get("synergy_weight", 0.0)),
        counter_weight=float(coefficients.get("counter_weight", 0.0)),
        role_completion_weight=float(coefficients.get("role_completion_weight", 0.0)),
        redundancy_penalty_weight=float(coefficients.get("redundancy_penalty_weight", 0.0)),
        regularization_c=float(regularization_c),
    )


def _fit_slot_blend_profile(slot_frame: pd.DataFrame, slot_index: int) -> PickOrderProfile:
    training_frame = slot_frame.copy()
    for column_name in BLEND_FEATURE_COLUMNS:
        if column_name not in training_frame.columns:
            training_frame[column_name] = 0.0

    X = training_frame[list(BLEND_FEATURE_COLUMNS)].fillna(0.0)
    y = training_frame["label_is_pick_fit"].astype(int)
    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)
    if positive_count <= 1 or negative_count <= 1:
        raise ValueError(f"Not enough labeled rows to fit pick slot blend profile {slot_index}.")

    cv_folds = min(5, positive_count, negative_count)
    if cv_folds < 2:
        cv_folds = 2

    model = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=cv_folds,
        class_weight="balanced",
        max_iter=2000,
        scoring="neg_log_loss",
        solver="lbfgs",
    )
    model.fit(X, y)

    coefficient_values = model.coef_[0].tolist()
    coefficients = {
        "base_score_weight": float(coefficient_values[0]),
        "secure_power_weight": float(coefficient_values[1]),
        "flexibility_weight": float(coefficient_values[2]),
        "synergy_weight": float(coefficient_values[3]),
        "counter_weight": float(coefficient_values[4]),
        "role_completion_weight": float(coefficient_values[5]),
        "redundancy_penalty_weight": float(coefficient_values[6]),
    }
    return _row_to_profile(
        slot_index=slot_index,
        intercept=float(model.intercept_[0]),
        coefficients=coefficients,
        regularization_c=float(model.C_[0]),
    )


def train_pick_order_profiles(training_frame: pd.DataFrame) -> PickOrderProfilesArtifact:
    if training_frame.empty:
        raise ValueError("Cannot train pick order profiles on an empty training frame.")

    fitted_profiles: dict[str, PickOrderProfilePayload] = {}
    global_profile = _fit_slot_blend_profile(training_frame, slot_index=0)
    for slot_index in sorted(int(value) for value in training_frame["slot_index"].dropna().unique().tolist()):
        slot_frame = training_frame[training_frame["slot_index"] == slot_index].reset_index(drop=True)
        try:
            profile = _fit_slot_blend_profile(slot_frame, slot_index=slot_index)
        except ValueError:
            profile_id, profile_title = _slot_identity(slot_index)
            profile = PickOrderProfile(
                slot_index=slot_index,
                id=profile_id,
                title=profile_title,
                summary=global_profile.summary.replace("pick slot 0", f"pick slot {slot_index}"),
                intercept=global_profile.intercept,
                base_score_weight=global_profile.base_score_weight,
                secure_power_weight=global_profile.secure_power_weight,
                flexibility_weight=global_profile.flexibility_weight,
                synergy_weight=global_profile.synergy_weight,
                counter_weight=global_profile.counter_weight,
                role_completion_weight=global_profile.role_completion_weight,
                redundancy_penalty_weight=global_profile.redundancy_penalty_weight,
                regularization_c=global_profile.regularization_c,
            )
        fitted_profiles[str(slot_index)] = {
            "slot_index": int(profile.slot_index),
            "id": profile.id,
            "title": profile.title,
            "summary": profile.summary,
            "intercept": float(profile.intercept),
            "base_score_weight": float(profile.base_score_weight),
            "secure_power_weight": float(profile.secure_power_weight),
            "flexibility_weight": float(profile.flexibility_weight),
            "synergy_weight": float(profile.synergy_weight),
            "counter_weight": float(profile.counter_weight),
            "role_completion_weight": float(profile.role_completion_weight),
            "redundancy_penalty_weight": float(profile.redundancy_penalty_weight),
            "regularization_c": float(profile.regularization_c),
        }

    return {
        "source": "trained-logregcv-slot-blend",
        "blend_features": list(BLEND_FEATURE_COLUMNS),
        "slot_profiles": fitted_profiles,
    }


def _validate_pick_order_profiles(payload: Any) -> PickOrderProfilesArtifact | None:
    if not isinstance(payload, dict):
        return None
    slot_profiles = payload.get("slot_profiles")
    blend_features = payload.get("blend_features")
    if not isinstance(slot_profiles, dict) or not isinstance(blend_features, list):
        return None

    validated_profiles: dict[str, PickOrderProfilePayload] = {}
    for slot_key, profile in slot_profiles.items():
        if not isinstance(profile, dict):
            continue
        validated_profiles[str(slot_key)] = {
            "slot_index": int(profile.get("slot_index", int(slot_key))),
            "id": str(profile.get("id", _slot_identity(int(slot_key))[0])),
            "title": str(profile.get("title", _slot_identity(int(slot_key))[1])),
            "summary": str(profile.get("summary", "")),
            "intercept": float(profile.get("intercept", 0.0)),
            "base_score_weight": float(profile.get("base_score_weight", 0.0)),
            "secure_power_weight": float(profile.get("secure_power_weight", 0.0)),
            "flexibility_weight": float(profile.get("flexibility_weight", 0.0)),
            "synergy_weight": float(profile.get("synergy_weight", 0.0)),
            "counter_weight": float(profile.get("counter_weight", 0.0)),
            "role_completion_weight": float(profile.get("role_completion_weight", 0.0)),
            "redundancy_penalty_weight": float(profile.get("redundancy_penalty_weight", 0.0)),
            "regularization_c": float(profile.get("regularization_c", 0.0)),
        }

    return {
        "source": str(payload.get("source", "trained-logregcv-slot-blend")),
        "blend_features": [str(item) for item in blend_features],
        "slot_profiles": validated_profiles,
    }


@lru_cache(maxsize=1)
def load_pick_order_profiles(path: Path = PICK_ORDER_PROFILE_PATH) -> PickOrderProfilesArtifact:
    validated_payload = _validate_pick_order_profiles(load_json(path))
    if validated_payload is None:
        raise FileNotFoundError(f"Pick order profiles not found at {path}. Run train_pick_ranker_models.py first.")
    return validated_payload


def resolve_pick_order_profile(
    global_pick_index: int,
    phase_index: int,
    pick_order: int,
) -> PickOrderProfile:
    del global_pick_index
    del phase_index

    payload = load_pick_order_profiles()
    profile_payload = payload["slot_profiles"].get(str(int(pick_order)))
    if profile_payload is None:
        raise KeyError(f"No trained pick order profile found for pick slot {pick_order}.")
    return PickOrderProfile(**profile_payload)


def weighted_signal_average_for_profile(
    frame: pd.DataFrame,
    order_profile: PickOrderProfile,
) -> pd.Series:
    coefficients = {
        "secure_power_signal": abs(float(order_profile.secure_power_weight)),
        "flexibility_signal": abs(float(order_profile.flexibility_weight)),
        "ally_pick_synergy_signal": abs(float(order_profile.synergy_weight)),
        "counter_vs_enemy_picks_signal": abs(float(order_profile.counter_weight)),
        "ally_role_completion_signal": abs(float(order_profile.role_completion_weight)),
    }
    total = sum(coefficients.values())
    if total <= 0:
        total = float(len(POSITIVE_SIGNAL_COLUMNS))
        coefficients = {column_name: 1.0 for column_name in POSITIVE_SIGNAL_COLUMNS}
    weighted_sum = sum(
        coefficients[column_name] * frame[column_name]
        for column_name in POSITIVE_SIGNAL_COLUMNS
    )
    return weighted_sum / total


def score_pick_order_profile(
    frame: pd.DataFrame,
    order_profile: PickOrderProfile,
) -> pd.Series:
    return (
        float(order_profile.intercept)
        + float(order_profile.base_score_weight) * frame["prior_score"]
        + float(order_profile.secure_power_weight) * frame["secure_power_signal"]
        + float(order_profile.flexibility_weight) * frame["flexibility_signal"]
        + float(order_profile.synergy_weight) * frame["ally_pick_synergy_signal"]
        + float(order_profile.counter_weight) * frame["counter_vs_enemy_picks_signal"]
        + float(order_profile.role_completion_weight) * frame["ally_role_completion_signal"]
        + float(order_profile.redundancy_penalty_weight) * frame["redundancy_penalty"]
    )


def score_pick_order_profiles_frame(
    frame: pd.DataFrame,
    profiles_artifact: PickOrderProfilesArtifact,
) -> pd.Series:
    scored = pd.Series(0.0, index=frame.index, dtype=float)
    for slot_key, profile_payload in profiles_artifact["slot_profiles"].items():
        slot_index = int(slot_key)
        mask = pd.to_numeric(frame.get("slot_index", -1), errors="coerce").fillna(-1).astype(int) == slot_index
        if not mask.any():
            continue
        profile = PickOrderProfile(**profile_payload)
        scored.loc[mask] = score_pick_order_profile(frame.loc[mask], profile)
    return scored
