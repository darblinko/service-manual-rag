"""Load and validate a golden question set."""

import json
from pathlib import Path

REQUIRED_FIELDS = ("id", "question", "ground_truth")


def load_golden_set(path: Path | str) -> list[dict]:
    """Read a JSON array of cases; each needs id, question and ground_truth."""
    path = Path(path)
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must hold a non-empty JSON array")
    for index, case in enumerate(cases):
        missing = [field for field in REQUIRED_FIELDS if not case.get(field)]
        if missing:
            raise ValueError(f"{path} case {index} is missing {', '.join(missing)}")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate case ids")
    return cases
