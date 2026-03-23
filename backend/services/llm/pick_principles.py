from __future__ import annotations

PICK_DRAFT_PRINCIPLES: list[dict[str, str]] = [
    {
        "id": "secure_power",
        "title": "Secure Power Early",
        "text": (
            "When the board is still open, strong global power and contest rate matter. "
            "High-power heroes are safer early picks when no sharper synergy or counter signal has appeared yet."
        ),
    },
    {
        "id": "build_ally_synergy",
        "title": "Build Ally Synergy",
        "text": (
            "If your side has already revealed key picks, move up heroes that pair well with them. "
            "A clean synergy chain is one of the clearest reasons to prefer one viable pick over another."
        ),
    },
    {
        "id": "pressure_enemy_reveals",
        "title": "Pressure Enemy Reveals",
        "text": (
            "When the enemy has shown important heroes, prioritize picks that statistically perform well into them. "
            "Strong counter pressure can be worth more than a generic comfort or power pick."
        ),
    },
    {
        "id": "complete_missing_roles",
        "title": "Complete Missing Roles",
        "text": (
            "As your composition fills out, value heroes that cleanly solve an unfilled role. "
            "Role completion becomes stronger when it also keeps the overall team functional and easy to execute."
        ),
    },
    {
        "id": "preserve_flexibility",
        "title": "Preserve Flexibility",
        "text": (
            "Flexible heroes help hide your final structure and keep later draft branches open. "
            "When two options are close in power, flexibility is a meaningful tiebreaker."
        ),
    },
    {
        "id": "avoid_redundancy",
        "title": "Avoid Redundancy",
        "text": (
            "Do not over-index on one role or function too early unless the payoff is exceptional. "
            "Redundant picks can make later role completion harder and narrow your draft unnecessarily."
        ),
    },
    {
        "id": "fit_over_raw_power",
        "title": "Fit Over Raw Power",
        "text": (
            "A slightly lower-power hero can still be the better pick if it fits your revealed allies, "
            "punishes the enemy draft, or solves an urgent role gap."
        ),
    },
    {
        "id": "keep_backups",
        "title": "Keep Backups",
        "text": (
            "Your second and third pick options should usually preserve the same draft goal. "
            "If the best option disappears, the backup should still maintain synergy, counter pressure, or role coverage."
        ),
    },
]
