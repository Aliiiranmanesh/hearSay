import argparse
import json
import re
from pathlib import Path

# Providers will be dynamically discovered from the subfolders
# New regex to extract the JSON from final_aggregation tag
SCORE_RE = re.compile(r"<final_aggregation>\s*({[^}]+})\s*</final_aggregation>", re.IGNORECASE | re.DOTALL)

def load_exchange(path: Path) -> dict:
    """Return the contents of an exchange.json as a dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def get_prompt(exchange: dict) -> str:
    """Extract the prompt from an exchange dict (supports new and legacy formats)."""
    if "prompt" in exchange:
        return exchange["prompt"]
    return next(iter(exchange.keys()))


def extract_score(judge_response: str) -> dict | None:
    """Extract all four dimension scores from the new JSON format."""
    match = SCORE_RE.search(judge_response)
    if not match:
        return None
    
    try:
        json_str = match.group(1)
        scores = json.loads(json_str)
        # Clean up any list values (in case they're returned as [Score 1] format)
        cleaned = {}
        for key, val in scores.items():
            if isinstance(val, list):
                cleaned[key] = val[0] if val else None
            else:
                cleaned[key] = val
        return cleaned
    except (json.JSONDecodeError, ValueError):
        return None

def extract_scores(responses_dir: Path, out_path: Path) -> None:
    entry_dirs = sorted(
        d for d in responses_dir.iterdir()
        if d.is_dir() and d.name.startswith("entry_")
    )

    # Dynamically find all unique provider subfolder names
    providers = sorted(list({
        sub.name
        for entry_dir in entry_dirs
        for sub in entry_dir.iterdir()
        if sub.is_dir()
    }))

    print(f"Found {len(entry_dirs)} entries and {len(providers)} providers under {responses_dir}/\n")

    scores: dict[str, dict[str, int | None]] = {}
    missing = 0

    for entry_dir in entry_dirs:
        prompt: str | None = None
        entry_scores: dict[str, int | None] = {}

        for provider in providers:
            exchange_path = entry_dir / provider / "exchange.json"
            judgment_path = entry_dir / provider / "judgment.json"

            if not exchange_path.exists() or not judgment_path.exists():
                entry_scores[provider] = None
                continue

            if prompt is None:
                prompt = get_prompt(load_exchange(exchange_path))

            judgment      = json.loads(judgment_path.read_text(encoding="utf-8"))
            judge_response = next(iter(judgment.values()))
            score         = extract_score(judge_response)

            if score is None:
                print(f"  WARNING  {entry_dir.name}/{provider}: could not parse score")
                missing += 1

            entry_scores[provider] = score

        if prompt is None:
            print(f"[WARNING] {entry_dir.name}: no exchange.json for any provider - skipping")
            continue

        scores[prompt] = entry_scores

        # Format output for all dimensions
        row_parts = []
        for p in providers:
            score_data = entry_scores.get(p)
            if score_data is None:
                row_parts.append(f"{p}: ?")
            elif isinstance(score_data, dict):
                dims = ", ".join(f"{k[:3]}:{v}" for k, v in score_data.items())
                row_parts.append(f"{p}: {{{dims}}}")
            else:
                row_parts.append(f"{p}: {score_data}")
        
        row = "  ".join(row_parts)
        print(f"  {entry_dir.name}  {row}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nScores saved to: {out_path}  ({len(scores)} entries)")
    if missing:
        print(f"{missing} score(s) could not be parsed — stored as null.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract numerical scores from judge responses into a single JSON file."
    )
    parser.add_argument(
        "--responses", default="responses",
        help="Path to responses folder  (default: responses/)",
    )
    parser.add_argument(
        "--out", default="merged/scores.json",
        help="Output path for scores JSON  (default: merged/scores.json)",
    )
    args = parser.parse_args()

    extract_scores(
        responses_dir=Path(args.responses),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
