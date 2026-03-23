from __future__ import annotations

DRAFT_PRINCIPLES: list[dict[str, str]] = [
    {
        "id": "early_meta_pressure",
        "title": "Early Meta Pressure",
        "text": (
            "In the first ban phase, prioritize heroes that are globally contested and repeatedly removed early. "
            "When a hero has both strong ban presence and strong early-phase fit, denying it now is usually better "
            "than saving the ban for a narrow counter later."
        ),
    },
    {
        "id": "slot_specific_timing",
        "title": "Slot-Specific Timing",
        "text": (
            "A hero that historically aligns with the exact current ban slot should be valued more. "
            "Current-slot fit matters because some heroes are consistently removed immediately while others are "
            "better held for later bans."
        ),
    },
    {
        "id": "deny_enemy_synergy",
        "title": "Deny Enemy Synergy",
        "text": (
            "If the enemy has already revealed a core pick, ban heroes that pair too cleanly with that core. "
            "Strong synergy with revealed enemy picks is one of the clearest reasons to move a ban upward."
        ),
    },
    {
        "id": "deny_counter_threat",
        "title": "Protect Revealed Core",
        "text": (
            "When your side has already shown heroes, prioritize banning candidates that strongly counter those "
            "revealed picks. Preventing a hard punish can matter more than removing a generic high-power hero."
        ),
    },
    {
        "id": "deny_role_completion",
        "title": "Deny Role Completion",
        "text": (
            "If a candidate cleanly fills one of the enemy's missing roles, that candidate becomes a stronger ban. "
            "Bans that stop clean role completion force the enemy into weaker or less flexible follow-up picks."
        ),
    },
    {
        "id": "respect_flexibility",
        "title": "Respect Flexibility",
        "text": (
            "Flexible heroes are more dangerous in draft because they hide the enemy's final lane and role structure. "
            "When flexibility is paired with power or contested status, it becomes a stronger ban signal."
        ),
    },
    {
        "id": "pattern_continuation",
        "title": "Ban Pattern Continuation",
        "text": (
            "A good next ban often matches the type of heroes already being targeted. "
            "Similarity to prior bans can indicate that the current team is continuing a coherent denial plan."
        ),
    },
    {
        "id": "backup_adjustment",
        "title": "Backup Adjustment",
        "text": (
            "The second and third ban options matter because the state can change after each enemy move. "
            "Keep backups that preserve the same strategic goal: deny synergy, deny counters, or deny role completion."
        ),
    },
]

