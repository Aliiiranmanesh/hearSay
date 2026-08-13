import argparse
import json
from pathlib import Path

from run_batch import parse_entries, process_entries, load_entries_from_hf
from run_judge import run_judge
from merge import merge
from score_extract import extract_scores


def main():
    parser = argparse.ArgumentParser(description="Full HearSayBench pipeline.")
    parser.add_argument("input_file", nargs="?", default="aliIranmanesh/HearSayBench", help="Path to local file OR Hugging Face repo ID (default: aliIranmanesh/HearSayBench)")
    parser.add_argument("--out",    default="responses", help="Intermediate responses folder  (default: responses/)")
    parser.add_argument("--merged", default="merged",    help="Final merged output folder     (default: merged/)")
    parser.add_argument("--model",  default="gemini-3.5-flash",    help="Judge model (default: gemini-3.5-flash)")
    parser.add_argument("--delay",  default=1.5, type=float, help="Seconds between API calls (default: 1.5)")
    parser.add_argument("--start",  default=1,   type=int,   help="Resume batch from entry N  (default: 1)")
    parser.add_argument("--end",    default=None, type=int,   help="Limit batch to entry N     (default: None)")
    parser.add_argument("--steps",  default="all",        help="Steps to run (all, responses, judge, harm, merge, scores). Use comma for multiple.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    responses_dir = Path(args.out)
    merged_dir    = Path(args.merged)
    steps = [s.strip() for s in args.steps.split(",")]

    # Check if input_file is a local file or Hugging Face repo ID
    if input_path.exists():
        print(f"Loading local dataset file: {input_path}")
        text = input_path.read_text(encoding="utf-8")
        entries = parse_entries(text)
    else:
        # Load from Hugging Face Hub
        repo_id = args.input_file
        print(f"Loading benchmark dataset from Hugging Face: {repo_id}")
        entries = load_entries_from_hf(repo_id)

    # Slice entries based on start/end
    start_idx = args.start - 1
    end_idx = args.end if args.end is not None else len(entries)
    entries = entries[start_idx:end_idx]

    # 1: Get model responses
    if "all" in steps or "responses" in steps:
        print("=" * 60)
        print("STEP 1 — Collecting model responses")
        print("=" * 60)
        print(f"Processing {len(entries)} entries from {args.input_file}")
        process_entries(entries, out_dir=responses_dir, delay=args.delay, start_idx=args.start)


    # 2: Judge all outlines
    if "all" in steps or "judge" in steps:
        print("\n" + "=" * 60)
        print("STEP 2 — Judging responses")
        print("=" * 60)
        
        # Regenerate outlines first to ensure they use the latest EVALUATION_TEMPLATE
        print("Regenerating outline.txt files with latest template...")
        from evaluator import EVALUATION_TEMPLATE
        
        target_entry_dirs = sorted(responses_dir.glob("entry_*"))
        if args.end is not None:
            target_entry_dirs = target_entry_dirs[:args.end]
        if args.start > 1:
            target_entry_dirs = target_entry_dirs[args.start-1:]
            
        for entry_dir in target_entry_dirs:
            rw_path = entry_dir / "real_world.txt"
            if not rw_path.exists(): continue
            real_world = rw_path.read_text(encoding="utf-8").strip()
            
            for provider_dir in entry_dir.iterdir():
                if not provider_dir.is_dir(): continue
                ex_path = provider_dir / "exchange.json"
                if not ex_path.exists(): continue
                
                ex = json.loads(ex_path.read_text(encoding="utf-8"))
                prompt = ex.get("prompt") or next(iter(ex.keys()))
                model_res = ex.get("model_response") or ex.get(prompt)
                
                if not model_res: continue
                
                outline_text = EVALUATION_TEMPLATE.format(
                    real_world=real_world,
                    prompt=prompt,
                    model_response=model_res
                ).strip()
                
                (provider_dir / "outline.txt").write_text(outline_text, encoding="utf-8")
                
                # If we're forcing a re-judge, delete old judgment
                # judgment_path = provider_dir / "judgment.json"
                # if judgment_path.exists(): judgment_path.unlink()

        run_judge(responses_dir=responses_dir, model=args.model, delay=args.delay, end=args.end)

    # 2.5: Evaluate harmfulness of responses
    if "all" in steps or "harm" in steps:
        print("\n" + "=" * 60)
        print("STEP 2.5 - Evaluating response harmfulness")
        print("=" * 60)
        from harm_eval import run_harm_eval
        run_harm_eval(
            responses_dir=responses_dir,
            providers=[],  # Auto-discovers all active provider folders
            model=args.model,
            delay=args.delay,
            limit=args.end,
            start_idx=args.start,
        )

    # 3: Merge into final files
    if "all" in steps or "merge" in steps:
        print("\n" + "=" * 60)
        print("STEP 3 — Merging into final files")
        print("=" * 60)
        merge(responses_dir=responses_dir, out_dir=merged_dir)

    # 4: Extract scores
    if "all" in steps or "scores" in steps:
        print("\n" + "=" * 60)
        print("STEP 4 — Extracting judge scores")
        print("=" * 60)
        extract_scores(responses_dir=responses_dir, out_path=merged_dir / "scores.json",)
        
        print("\n" + "=" * 60)
        print("STEP 5 — Appending harm evaluations to merged files")
        print("=" * 60)
        from add_harm_to_merged import add_harm_to_merged
        add_harm_to_merged(responses_dir=responses_dir, merged_dir=merged_dir)

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Final files in: {merged_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
