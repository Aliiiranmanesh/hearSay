import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_client import Message, OpenAIProvider, GeminiProvider

JUDGE_SYSTEM_PROMPT = "You are a rigorous and fair NLP benchmark evaluator."

JUDGE_LOCK = threading.Lock()
JUDGE_LAST_CALLED = [0.0]
CONSOLE_LOCK = threading.Lock()


def process_single_judgment(
    idx: int,
    total: int,
    outline_path: Path,
    responses_dir: Path,
    judge,
    delay: float,
) -> None:
    judgment_path = outline_path.parent / "judgment.json"

    # Skip if already judged
    if judgment_path.exists():
        with CONSOLE_LOCK:
            print(f"[{idx}/{total}] {outline_path.parent.relative_to(responses_dir)} - already done, skipping")
        return

    # Pacing delay
    if delay > 0:
        with JUDGE_LOCK:
            now = time.time()
            elapsed = now - JUDGE_LAST_CALLED[0]
            if elapsed < delay:
                time.sleep(delay - elapsed)
            JUDGE_LAST_CALLED[0] = time.time()

    with CONSOLE_LOCK:
        print(f"[{idx}/{total}] {outline_path.parent.relative_to(responses_dir)} ... calling...")

    judge_prompt = outline_path.read_text(encoding="utf-8").strip()

    try:
        response = judge.call(
            [
                Message(role="system", content=JUDGE_SYSTEM_PROMPT),
                Message(role="user",   content=judge_prompt),
            ],
            temperature=0.2,
            max_tokens=16384,
        )
        judge_response = response.content

        judgment_path.write_text(
            json.dumps({judge_prompt: judge_response}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with CONSOLE_LOCK:
            print(f"[{idx}/{total}] {outline_path.parent.relative_to(responses_dir)} ... OK")

    except Exception as exc:
        with CONSOLE_LOCK:
            print(f"[{idx}/{total}] {outline_path.parent.relative_to(responses_dir)} ... ERROR - {exc}")
        (outline_path.parent / "judge_error.txt").write_text(str(exc), encoding="utf-8")


def run_judge(responses_dir: Path, model: str, delay: float, end: int | None = None) -> None:
    if "gemini" in model.lower():
        judge = GeminiProvider(model=model)
    else:
        judge = OpenAIProvider(model=model)

    outlines = sorted(responses_dir.glob("entry_*/" + "*/outline.txt"))
    
    if end is not None:
        # Filter to only include the first 'end' entries
        # entry_XXXX directories are sorted, so we can just pick the ones we want
        entry_dirs = sorted(responses_dir.glob("entry_*"))
        target_entry_names = {d.name for d in entry_dirs[:end]}
        outlines = [o for o in outlines if o.parts[-3] in target_entry_names]

    if not outlines:
        print(f"No outline.txt files found under {responses_dir}/")
        return

    total = len(outlines)
    print(f"Found {total} judge prompts under {responses_dir}/\n")
    print(f"Judging {total} prompts in parallel using ThreadPool...")

    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_judgment,
                idx,
                total,
                outline_path,
                responses_dir,
                judge,
                delay
            )
            for idx, outline_path in enumerate(outlines, start=1)
        ]

        # Block and wait for all judging tasks to complete
        for future in futures:
            future.result()

    print(f"\nJudging complete. Results saved alongside each outline.txt as judgment.json")


def main():
    parser = argparse.ArgumentParser(description="Judge all outline prompts with an OpenAI model.")
    parser.add_argument("--responses", default="responses", help="Path to responses folder  (default: responses/)")
    parser.add_argument("--model",     default="gemini-2.5-flash",    help="Judge model (default: gemini-2.5-flash)")
    parser.add_argument("--delay",     default=1.0, type=float, help="Seconds between calls  (default: 1.0)")
    parser.add_argument("--end",       default=None, type=int,  help="Limit to first N entries")
    args = parser.parse_args()

    responses_dir = Path(args.responses)
    if not responses_dir.exists():
        raise FileNotFoundError(f"Responses folder not found: {responses_dir}")

    run_judge(responses_dir=responses_dir, model=args.model, delay=args.delay, end=args.end)


if __name__ == "__main__":
    main()
