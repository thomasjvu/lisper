#!/usr/bin/env python3
"""Build a local publish verdict from a downloaded Lisper eval artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
DEFAULT_EVAL_JSON = (
    REPO_ROOT / "data" / "processed" / "gemma4_audio" / "artifacts" / "eval_v4" / "tuned_eval.json"
)
ALLOWED_CLASSES = {"clear", "frontal", "lateral", "dental", "palatal"}
EXPECTED_LABELS = ("Detected class", "Reason", "Corrective cue", "Encouragement")
FIELD_PATTERNS = {
    "detected class": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Detected class\s*:\s*(.+)$", re.IGNORECASE),
    "reason": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Reason\s*:\s*(.+)$", re.IGNORECASE),
    "corrective cue": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Corrective cue\s*:\s*(.+)$", re.IGNORECASE),
    "encouragement": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Encouragement\s*:\s*(.+)$", re.IGNORECASE),
}
METRIC_KEYS = (
    "class_match",
    "clear_match",
    "has_reason",
    "has_corrective_cue",
    "has_encouragement",
    "format_exact",
    "format_four_lines",
    "detected_class_in_schema",
)


def normalize_detected_class(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.strip("[]*`")
    return re.sub(r"\s+", " ", normalized)


def extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        for key, pattern in FIELD_PATTERNS.items():
            match = pattern.match(raw_line.rstrip())
            if match:
                fields[key] = match.group(1).strip().strip("[]")
                break
    return fields


def score_format(text: str) -> dict[str, bool]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    format_four_lines = len(lines) == 4
    format_exact = format_four_lines
    if format_exact:
        for line, label in zip(lines, EXPECTED_LABELS):
            prefix = f"{label}:"
            if not line.startswith(prefix) or not line[len(prefix) :].strip():
                format_exact = False
                break
    return {"format_four_lines": format_four_lines, "format_exact": format_exact}


def score_response(row: dict) -> dict[str, bool]:
    response = str(row.get("response") or "")
    expected = str(row.get("lisp_type") or "").lower()
    fields = extract_fields(response)
    detected = normalize_detected_class(fields.get("detected class", ""))
    return {
        "class_match": detected == expected,
        "clear_match": (detected == "clear") == (expected == "clear"),
        "has_reason": bool(fields.get("reason")),
        "has_corrective_cue": bool(fields.get("corrective cue")),
        "has_encouragement": bool(fields.get("encouragement")),
        "detected_class_in_schema": detected in ALLOWED_CLASSES,
        **score_format(response),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    scored_rows = []
    for row in rows:
        metrics = score_response(row) if row.get("error") is None else {key: False for key in METRIC_KEYS}
        scored_rows.append({**row, **metrics})

    successful_rows = [row for row in scored_rows if row.get("error") is None]
    summary = {
        "count": len(scored_rows),
        "success_count": len(successful_rows),
        "effective_success_count": len(successful_rows),
        "error_count": len(scored_rows) - len(successful_rows),
        "hard_error_count": len(scored_rows) - len(successful_rows),
        "hard_error_ids": [row["id"] for row in scored_rows if row.get("error") is not None],
        "truncated_count": sum(1 for row in scored_rows if row.get("used_truncation")),
        "in_memory_retry_count": sum(1 for row in scored_rows if row.get("used_in_memory_audio")),
    }
    for key in METRIC_KEYS:
        summary[key] = sum(1 for row in scored_rows if row[key]) / max(len(scored_rows), 1)
        summary[f"{key}_successful_only"] = sum(1 for row in successful_rows if row[key]) / max(
            len(successful_rows), 1
        )
    return summary


def build_publish_verdict(
    summary: dict,
    adapter_root: str | None,
    bundle_path: str | None,
    eval_limit: int | None,
    min_class_match: float,
    min_clear_match: float,
    min_format_exact: float,
    min_encouragement: float,
) -> dict:
    thresholds = {
        "min_class_match_successful_only": min_class_match,
        "min_clear_match_successful_only": min_clear_match,
        "min_format_exact_successful_only": min_format_exact,
        "min_has_encouragement_successful_only": min_encouragement,
        "require_zero_hard_errors": True,
    }
    reasons = []
    if summary["hard_error_count"] != 0:
        reasons.append(f"hard_error_count={summary['hard_error_count']} must be 0")
    if summary["class_match_successful_only"] < min_class_match:
        reasons.append(f"class_match_successful_only={summary['class_match_successful_only']:.4f} < {min_class_match:.4f}")
    if summary["clear_match_successful_only"] < min_clear_match:
        reasons.append(f"clear_match_successful_only={summary['clear_match_successful_only']:.4f} < {min_clear_match:.4f}")
    if summary["format_exact_successful_only"] < min_format_exact:
        reasons.append(f"format_exact_successful_only={summary['format_exact_successful_only']:.4f} < {min_format_exact:.4f}")
    if summary["has_encouragement_successful_only"] < min_encouragement:
        reasons.append(
            f"has_encouragement_successful_only={summary['has_encouragement_successful_only']:.4f} < {min_encouragement:.4f}"
        )
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "thresholds": thresholds,
        "adapter_root": adapter_root,
        "bundle_path": bundle_path,
        "eval_limit": eval_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local publish_verdict.json from tuned_eval.json")
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-class-match", type=float, default=0.60)
    parser.add_argument("--min-clear-match", type=float, default=0.90)
    parser.add_argument("--min-format-exact", type=float, default=0.95)
    parser.add_argument("--min-encouragement", type=float, default=0.90)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"No rows array found in {args.eval_json}")

    summary = aggregate_metrics(rows)
    verdict = build_publish_verdict(
        summary=summary,
        adapter_root=payload.get("adapter_root"),
        bundle_path=payload.get("bundle_path"),
        eval_limit=summary["count"],
        min_class_match=args.min_class_match,
        min_clear_match=args.min_clear_match,
        min_format_exact=args.min_format_exact,
        min_encouragement=args.min_encouragement,
    )
    output_path = args.output or args.eval_json.with_name("publish_verdict.json")
    output_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(output_path), "status": verdict["status"], "reasons": verdict["reasons"]}, indent=2))


if __name__ == "__main__":
    main()
