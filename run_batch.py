import argparse
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluator import EVALUATION_TEMPLATE
from llm_client import (
    GeminiProvider, 
    Message, 
    OpenAIProvider, 
    QwenProvider, 
    LlamaProvider,
    DeepSeekProvider,
    GptOssProvider,
    GemmaProvider,
    KimiProvider,
    LfmProvider,
    AnthropicProvider,
)

PROVIDERS: dict[str, object] = {
    # OpenAI
    "gpt-5.5":              OpenAIProvider(model="gpt-5.5"),
    
    # Gemini
    "gemini-3-flash":       GeminiProvider(model="gemini-3-flash-preview"),
    
    # Llama (via TogetherAI)
    "llama":                LlamaProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    
    # Qwen (via TogetherAI)
    "qwen-3.6-plus":        QwenProvider(model="Qwen/Qwen3.6-Plus"),
    
    # DeepSeek (via TogetherAI)
    "deepseek-v4":          DeepSeekProvider(model="deepseek-ai/DeepSeek-V4-Pro"),
    
    # GPT-OSS (via TogetherAI)
    "gpt-oss-120b":         GptOssProvider(model="openai/gpt-oss-120b"),
    "gpt-oss-20b":          GptOssProvider(model="openai/gpt-oss-20b"),
    
    # Gemma (via TogetherAI)
    "gemma-4":              GemmaProvider(model="google/gemma-4-31B-it"),
    
    # Kimi (via TogetherAI)
    "kimi-k2.6":            KimiProvider(model="moonshotai/Kimi-K2.6"),
    
    # LFM (via TogetherAI)
    "lfm2-24b":             LfmProvider(model="LiquidAI/LFM2-24B-A2B"),
    
    # Anthropic
    "claude-sonnet-4-6":    AnthropicProvider(model="claude-sonnet-4-6"),
}

PROVIDER_DELAYS: dict[str, float] = {
    "gpt-5.5":              1.5,
    "gemini-3-flash":       60.0,
    "llama":                1.5,
    "qwen-3.6-plus":        1.5,
    "deepseek-v4":          1.5,
    "gpt-oss-120b":         1.5,
    "gpt-oss-20b":          1.5,
    "gemma-4":              1.5,
    "kimi-k2.6":            1.5,
    "lfm2-24b":             1.5,
    "claude-sonnet-4-6":    1.5,
}



# Concurrency Locks and Tracking for safe execution
PROVIDER_LOCKS: dict[str, threading.Lock] = {name: threading.Lock() for name in PROVIDERS}
PROVIDER_LAST_CALLED: dict[str, float] = {name: 0.0 for name in PROVIDERS}
CONSOLE_LOCK = threading.Lock()

def parse_entries(text: str) -> list[tuple[str, str, str]]:
    """
    Parse the numbered file format and return a list of (scenario, real_world, prompt) triples.

    Each numbered block has the structure:
        N. <scenario — the title line>
        <real_world — one or more lines>
        —
        <prompt — one or more lines until the next numbered entry>
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    blocks = re.split(r"\n(?=\d+\.\s)", text.strip())

    entries = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        scenario_match = re.match(r"^\d+\.\s+(.+)", block)
        if not scenario_match:
            print(f"Skipping malformed block (no number): {block[:60]!r}…")
            continue
        scenario = scenario_match.group(1).split("\n")[0].strip()

        rest = block.split("\n", 1)[1].strip() if "\n" in block else ""

        parts = re.split(r"\n\s*\u2014\s*\n", rest, maxsplit=1)
        if len(parts) < 2:
            print(f"Skipping malformed block (no '—' separator): {scenario[:60]!r}…")
            continue

        real_world = parts[0].strip()
        prompt     = parts[1].strip()

        if scenario and real_world and prompt:
            entries.append((scenario, real_world, prompt))
        else:
            print(f"Skipping malformed block: {scenario[:60]!r}…")

    return entries

def call_provider(provider, prompt: str) -> str:
    response = provider.call([Message(role="user", content=prompt)], max_tokens=4096)
    return response.content


def save_entry(
    entry_dir: Path,
    provider_name: str,
    prompt: str,
    model_response: str,
    real_world: str,
    scenario: str,
) -> None:
    provider_dir = entry_dir / provider_name
    provider_dir.mkdir(parents=True, exist_ok=True)

    (provider_dir / "exchange.json").write_text(
        json.dumps(
            {
                "scenario":       scenario,
                "prompt":         prompt,
                "model_response": model_response,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outline = EVALUATION_TEMPLATE.format(
        real_world=real_world,
        prompt=prompt,
        model_response=model_response,
    )
    (provider_dir / "outline.txt").write_text(outline, encoding="utf-8")


def process_single_task(
    idx: int,
    total: int,
    label: str,
    entry_dir: Path,
    scenario: str,
    real_world: str,
    prompt: str,
    provider_name: str,
    provider,
    delay: float,
) -> None:
    exchange_path = entry_dir / provider_name / "exchange.json"
    if exchange_path.exists():
        with CONSOLE_LOCK:
            print(f"[{idx}/{total}] [{label}] -> {provider_name:<15} already done, skipping")
        return

    # Safe pacing lock
    cooldown = PROVIDER_DELAYS.get(provider_name, delay)
    with PROVIDER_LOCKS[provider_name]:
        now = time.time()
        elapsed = now - PROVIDER_LAST_CALLED[provider_name]
        if elapsed < cooldown:
            time.sleep(cooldown - elapsed)
        PROVIDER_LAST_CALLED[provider_name] = time.time()

    with CONSOLE_LOCK:
        print(f"[{idx}/{total}] [{label}] -> {provider_name:<15} calling...")

    try:
        model_response = call_provider(provider, prompt)
        save_entry(entry_dir, provider_name, prompt, model_response, real_world, scenario)
        with CONSOLE_LOCK:
            print(f"[{idx}/{total}] [{label}] -> {provider_name:<15} ✓")
    except Exception as exc:
        with CONSOLE_LOCK:
            print(f"[{idx}/{total}] [{label}] -> {provider_name:<15} ERROR — {exc}")
        err_path = entry_dir / provider_name
        err_path.mkdir(exist_ok=True, parents=True)
        (err_path / "error.txt").write_text(str(exc), encoding="utf-8")


def process_entries(
    entries: list[tuple[str, str, str]],
    out_dir: Path,
    delay: float = 1.5,
    start_idx: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries) + start_idx - 1

    # Initialize directories and text files synchronously to avoid race conditions on file operations
    tasks = []
    for idx, (scenario, real_world, prompt) in enumerate(entries, start=start_idx):
        label     = f"entry_{idx:04d}"
        entry_dir = out_dir / label
        entry_dir.mkdir(exist_ok=True)

        scenario_path = entry_dir / "scenario.txt"
        if not scenario_path.exists():
            scenario_path.write_text(scenario, encoding="utf-8")

        rw_path = entry_dir / "real_world.txt"
        if not rw_path.exists():
            rw_path.write_text(real_world, encoding="utf-8")

        for provider_name, provider in PROVIDERS.items():
            tasks.append((idx, total, label, entry_dir, scenario, real_world, prompt, provider_name, provider))

    # Run remaining tasks concurrently
    print(f"Executing {len(tasks)} model-scenario tasks in parallel using ThreadPool...")
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_task,
                idx,
                total,
                label,
                entry_dir,
                scenario,
                real_world,
                prompt,
                provider_name,
                provider,
                delay
            )
            for idx, total, label, entry_dir, scenario, real_world, prompt, provider_name, provider in tasks
        ]
        
        # Block and wait for all tasks to complete
        for future in futures:
            future.result()

    print(f"\nFinished. Results saved to: {out_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Batch-process HearSayBench entries through all providers.")
    parser.add_argument("input_file",        help="Path to the numbered entries file")
    parser.add_argument("--out",   default="responses", help="Output directory  (default: responses/)")
    parser.add_argument("--delay", default=1.5, type=float, help="Seconds between API calls  (default: 1.5)")
    parser.add_argument("--start", default=1,   type=int,   help="Resume from entry N  (1-indexed)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text    = input_path.read_text(encoding="utf-8")
    entries = parse_entries(text)
    print(f"Parsed {len(entries)} entries from {input_path.name}")

    if args.start > 1:
        entries = entries[args.start - 1:]
        print(f"  Resuming from entry {args.start}")

    print(f"  Providers: {', '.join(PROVIDERS)}\n")
    process_entries(entries, out_dir=Path(args.out), delay=args.delay)


if __name__ == "__main__":
    main()
