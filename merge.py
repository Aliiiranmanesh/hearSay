import argparse
import json
from pathlib import Path

# Providers will be dynamically discovered from the subfolders


def load_exchange(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_first_value(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return next(iter(data.values()))


def merge(responses_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_dirs = sorted(d for d in responses_dir.iterdir() if d.is_dir() and d.name.startswith("entry_"))

    if not entry_dirs:
        print(f"No entry folders found under {responses_dir}/")
        return

    # Dynamically find all unique provider subfolder names
    providers = sorted(list({
        sub.name
        for entry_dir in entry_dirs
        for sub in entry_dir.iterdir()
        if sub.is_dir()
    }))

    print(f"Found {len(entry_dirs)} entries and {len(providers)} providers under {responses_dir}/\n")

    datasets: dict[str, dict] = {p: {} for p in providers}
    skipped = 0

    for entry_dir in entry_dirs:
        scenario_path = entry_dir / "scenario.txt"
        rw_path       = entry_dir / "real_world.txt"

        if not rw_path.exists():
            print(f"⚠  {entry_dir.name}: missing real_world.txt — skipping")
            skipped += 1
            continue

        real_world = rw_path.read_text(encoding="utf-8").strip()
        scenario   = scenario_path.read_text(encoding="utf-8").strip() if scenario_path.exists() else ""

        for provider in providers:
            exchange_path  = entry_dir / provider / "exchange.json"
            judgment_path  = entry_dir / provider / "judgment.json"

            missing = [p.name for p in [exchange_path, judgment_path] if not p.exists()]
            if missing:
                print(f"⚠  {entry_dir.name}/{provider}: missing {', '.join(missing)} — skipping")
                skipped += 1
                continue

            exchange = load_exchange(exchange_path)

            if "prompt" in exchange and "model_response" in exchange:
                prompt         = exchange["prompt"]
                model_response = exchange["model_response"]
                entry_scenario = exchange.get("scenario", scenario)
            else:
                prompt         = next(iter(exchange.keys()))
                model_response = next(iter(exchange.values()))
                entry_scenario = scenario

            judge_response = load_json_first_value(judgment_path)

            datasets[provider][prompt] = {
                "scenario":       entry_scenario,
                "model_response": model_response,
                "real_world":     real_world,
                "judge_response": judge_response,
            }

    # Write one file per provider
    for provider, data in datasets.items():
        out_path = out_dir / f"{provider}.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{provider:10} → {out_path}  ({len(data)} entries)")

    if skipped:
        print(f"\n {skipped} provider-entry pair(s) skipped due to missing files.")

    print("\nMerge complete.")


def main():
    parser = argparse.ArgumentParser(description="Merge batch results into three provider JSON files.")
    parser.add_argument("--responses", default="responses", help="Input responses folder  (default: responses/)")
    parser.add_argument("--out",       default="merged",    help="Output folder  (default: merged/)")
    args = parser.parse_args()

    merge(
        responses_dir=Path(args.responses),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
