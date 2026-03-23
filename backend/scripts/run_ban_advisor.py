from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.services.common.parser import parse_csv
from backend.services.modeling.advisor_pipeline import advise_bans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local ban recommendation + advisor pipeline without touching the app/frontend layer."
    )
    parser.add_argument("--team", default="blue", choices=["blue", "red"])
    parser.add_argument("--blue-picks", default="")
    parser.add_argument("--red-picks", default="")
    parser.add_argument("--blue-bans", default="")
    parser.add_argument("--red-bans", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--strict-turn", action="store_true")
    parser.add_argument("--rerank-pool-size", type=int, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON payload instead of only the advisor text.",
    )
    parser.add_argument(
        "--advisor-backend",
        default="tfidf",
        choices=["tfidf", "auto", "sentence-transformers"],
        help=(
            "Backend used for local principle retrieval. "
            "Defaults to tfidf to keep CLI output clean and avoid model-loading noise."
        ),
    )
    args = parser.parse_args()

    os.environ["LOCAL_DRAFT_ADVISOR_BACKEND"] = args.advisor_backend

    payload = advise_bans(
        team=args.team,
        blue_picks=parse_csv(args.blue_picks),
        red_picks=parse_csv(args.red_picks),
        blue_bans=parse_csv(args.blue_bans),
        red_bans=parse_csv(args.red_bans),
        top_k=args.top_k,
        strict_turn=args.strict_turn,
        rerank_pool_size=args.rerank_pool_size,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(payload["advisor"]["advice"])


if __name__ == "__main__":
    main()
