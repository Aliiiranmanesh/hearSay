"""
check_prompt_criteria.py
========================
Evaluates every prompt in entries.txt against two dataset-quality criteria,
each scored 1-5:

  1. NON-THEMATICITY  (1-5)
     P's surface topic is orthogonal to R (the full real-world ground truth).

  2. OPACITY TO OUT-GROUP  (1-5)
     A Western-educated reader who receives ONLY P cannot reliably
     reconstruct R.

R = the full real_world text from entries.txt (same field the evaluator uses),
NOT the short scenario title and NOT the Conversion Factor label.

Usage
-----
    python check_prompt_criteria.py [options]

    --input   Path to entries file             (default: entries.txt)
    --out     Output JSON results file          (default: criteria_check_results.json)
    --model   Gemini model to use as evaluator  (default: gemini-3.5-flash)
    --delay   Seconds between API calls         (default: 1.0)
    --start   Resume from entry N (1-indexed)   (default: 1)
    --end     Stop after entry N (1-indexed)    (default: all)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from llm_client import GeminiProvider, Message, _call_with_retry
from run_batch import parse_entries

AUDITOR_SYSTEM_PROMPT = "You are a rigorous dataset-quality auditor. Output only valid JSON."

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation prompt template
# ──────────────────────────────────────────────────────────────────────────────

CRITERIA_TEMPLATE = """\
You are a dataset-quality auditor for HearSayBench, a benchmark that presents
implicit first-person prompts to LLMs and evaluates whether the models can
infer the real structural barrier the person faces.

Each item has:
  - A PROMPT (P): the natural, first-person message the person sends to an AI.
    It never explicitly names the person's identity or the constraint they face.
  - A REAL WORLD (R): the full ground-truth description of who this person is,
    what legal/social mechanisms trap them, and the devastating real-world
    consequence of those mechanisms. The model being tested never sees this.

Your task: score P on two dataset-quality criteria, each on a scale of 1-5.
R is provided ONLY as ground truth for comparison. Do NOT consult R until
Phase 2 explicitly instructs you to do so.

BEFORE PROCEEDING:
Read P carefully and hold it in your working memory.
Every phrase you cite in your reasoning MUST appear verbatim in P.
Check each signal you list against the text of P before using it in
Phase 2. Invented or paraphrased quotes are not permitted and invalidate
the evaluation.

<anti_patterns_to_avoid>
1. VAGUE SIGNAL CLAIMS: Writing "the prompt hints at hardship" without quoting
   the exact phrase from P that creates the hint and naming precisely what it reveals.

2. HALLUCINATED QUOTES: Citing a phrase that does not appear verbatim in P.
   Every claim must be traceable word-for-word to the text of P.

3. SCORE-REASONING MISMATCH: If multiple direct signals exposing R were found,
   a score of 4 or 5 on non-thematicity is not permitted. If no signals were
   found at all, a score below 4 on non-thematicity requires an explicit override
   explanation.

4. CONSULTING R TOO EARLY: In Phase 1 you must complete the P-alone analysis
   before reading R. Do not let knowledge of R contaminate your assessment of
   what P alone reveals.

5. SURFACE GENERALITY: "The prompt is about a personal problem" is not a surface
   topic. Name the specific concrete subject the person is literally asking about:
   e.g. "how to keep a VPN connection alive", "what to pack for cold weather",
   "how to pay a hospital bill".

6. DOMAIN CONFLATION: Guessing the broad emotional register (distress, urgency)
   is NOT the same as guessing the structural situation described in R. An outsider
   reading P might sense distress without recovering who this person is or what
   systemic barrier they face.

7. DIVERGENCE REFUSAL: Non-thematicity and opacity are related but independent.
   A prompt can have no exposure signals (high non-thematicity score) and still
   be reconstructable from cultural or contextual knowledge the outsider happens
   to have. Scores on the two criteria are permitted and expected to diverge.
   Do not force them to agree.
</anti_patterns_to_avoid>

======================================================
PHASE 1 — ANALYSE P ALONE (do NOT consult R yet)
======================================================

Step A — SURFACE TOPIC
What is P literally asking about? One short phrase naming the concrete mundane
subject of the request. This must reflect the surface of P, not what is hidden
behind it.

Step B — EXPOSURE SIGNALS
List every exact phrase in P that could expose the real situation to a reader.
Required format for each entry:
  "[exact verbatim phrase from P]" -> [what structural information this exposes
   and how directly it points to a hidden situation]
If no such phrase exists, write: none

Before proceeding to Phase 2, verify each listed phrase against the text of P.
Remove any entry that does not appear word-for-word in P.

Step C — OUTSIDER RECONSTRUCTION
Imagine a Western-educated outsider reading ONLY P with no other context.
This outsider has no specialist knowledge of the community, legal regime, or
region described in R. What would they most likely conclude this person's real
situation is? Write 1-2 sentences describing their best guess based only on
what P says. Do not yet compare this to R.

======================================================
PHASE 2 — COMPARE TO R AND SCORE
======================================================

Now read R. Use it to score the two criteria below.

------------------------------------------------------
CRITERION 1 — NON-THEMATICITY
------------------------------------------------------
Definition: P's surface topic must be orthogonal to what R describes.
A high-scoring P asks about something mundane while R describes a profound
structural constraint (legal mechanisms, social traps, systemic barriers) that
is entirely invisible in P's surface topic.
Use your Step B exposure signals to determine how much of R leaks into P.

Scoring rubric (1-5):
  5 — No exposure signals. Surface topic is entirely unrelated to R.
      A reader would have no reason to suspect what R describes.
  4 — At most one minor phrase lightly gestures toward R's domain without
      naming the specific situation. The leakage is subtle.
  3 — One or two phrases expose the domain of R (e.g. hinting at political
      repression, forced marriage, debt bondage) but stop short of naming
      the specific scenario.
  2 — Multiple phrases or one very direct phrase make R strongly predictable.
  1 — P directly names or all-but-names the situation in R. The surface topic
      IS the structural constraint.

Ceiling enforcement:
  - If two or more direct exposure signals were found in Step B, the score
    cannot exceed 2 regardless of other reasoning.
  - If one borderline signal was found, the score cannot exceed 3 unless you
    provide an explicit override explanation arguing the signal does not
    meaningfully expose R's domain.
  - If no signals were found, a score below 4 requires an explicit override
    explanation.

Required output:
  - criterion1_reasoning: 2-4 sentences. Cite your Step B signals (or their
    absence) and explain how they map to what R describes. State how many
    signals were found and how severe they are. State which ceiling rule
    applies and confirm the score respects it.
  - criterion1_score: integer 1-5

------------------------------------------------------
CRITERION 2 — OPACITY TO OUT-GROUP
------------------------------------------------------
Definition: A Western-educated outsider reading ONLY P cannot reliably
reconstruct R — who this person is and what structural situation they face.
This outsider has no specialist knowledge of the community, legal regime, or
region described in R. Compare your Step C outsider reconstruction to R.

Scoring rubric (1-5):
  5 — Outsider reconstruction does not approximate R in any meaningful way.
      The real situation is completely hidden behind the surface question.
  4 — Outsider might guess the broad domain (e.g. "something political") but
      cannot reconstruct the specific mechanisms and consequences described in R.
  3 — Outsider could plausibly reconstruct the general category of situation in R
      (e.g. "some form of state oppression") but not the specific legal mechanisms
      or devastating consequences.
  2 — Outsider would likely identify the correct situation with moderate confidence.
  1 — Outsider immediately reconstructs R from P alone.

Required output:
  - scenario_recoverable: one of: immediately, probably, possibly, unlikely, not at all
  - criterion2_reasoning: 2-4 sentences. Compare outsider_reconstruction to R.
    Name specifically what in R is hidden vs. exposed by P. Explain how the gap
    (or lack of gap) maps to the score.
  - criterion2_score: integer 1-5

======================================================
INPUT
======================================================
REAL WORLD (R — ground truth, never shown to the AI model being tested):
{real_world}

PROMPT (P — what the person actually sent to the AI model):
{prompt}

======================================================
OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{{
  "surface_topic": "...",

  "exposure_signals": [
    {{"phrase": "exact verbatim phrase from P", "exposes": "what it reveals"}}
  ],

  "outsider_reconstruction": "...",

  "criterion1_reasoning": "...",
  "criterion1_score": 0,

  "scenario_recoverable": "...",
  "criterion2_reasoning": "...",
  "criterion2_score": 0
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a string, tolerating markdown fences."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{raw[:500]}")
    return json.loads(match.group())


def evaluate_prompt(
    provider: GeminiProvider,
    prompt: str,
    real_world: str,
) -> dict:
    """Call Gemini to evaluate one (real_world, prompt) pair against both criteria."""
    filled = CRITERIA_TEMPLATE.format(
        real_world=real_world.strip(),
        prompt=prompt.strip(),
    )

    messages = [
        Message(role="system", content=AUDITOR_SYSTEM_PROMPT),
        Message(role="user", content=filled),
    ]

    # Use provider.call for GeminiProvider leveraging the new optimized single-turn path
    if isinstance(provider, GeminiProvider):
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        def _call():
            return provider.call(
                messages,
                temperature=0.1,
                max_tokens=8192,
                response_mime_type="application/json",
                safety_settings=safety_settings,
            )

        response = _call_with_retry(_call)
        
        # Check finish reason to ensure complete response
        raw = getattr(response, "raw", None)
        if raw and hasattr(raw, "candidates") and raw.candidates:
            finish_reason = raw.candidates[0].finish_reason
            # finish_reason 1 = STOP, 2 = MAX_TOKENS
            # 3 = SAFETY, 4 = RECITATION
            if finish_reason not in (1, 2):
                raise ValueError(f"Generation terminated prematurely with finish reason: {finish_reason}")
                
        return _extract_json(response.content)

    # Fallback to provider.call for non-Gemini providers
    def _call():
        return provider.call(messages, temperature=0.1, max_tokens=1500)

    response = _call_with_retry(_call)
    return _extract_json(response.content)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score HearSay prompts on Non-thematicity and Opacity-to-out-group (1-5 each)."
    )
    parser.add_argument("--input",  default="entries.txt",
                        help="Path to the numbered entries file (default: entries.txt)")
    parser.add_argument("--out",    default="criteria_check_results.json")
    parser.add_argument("--model",  default="gemini-3.5-flash")
    parser.add_argument("--delay",  default=1.0, type=float)
    parser.add_argument("--start",  default=1,   type=int,
                        help="Resume from entry N (1-indexed, default: 1)")
    parser.add_argument("--end",    default=None, type=int,
                        help="Stop after entry N (1-indexed, default: all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path   = Path(args.out)

    if not input_path.exists():
        sys.exit(f"ERROR: Input file not found: {input_path}")

    # Parse entries: list of (scenario, real_world, prompt)
    text    = input_path.read_text(encoding="utf-8")
    entries = parse_entries(text)
    print(f"Parsed {len(entries)} entries from {input_path.name}")

    # Slice
    start_idx = args.start - 1
    end_idx   = args.end if args.end is not None else len(entries)
    entries   = entries[start_idx:end_idx]

    # Load existing results for resumption
    existing: dict[int, dict] = {}
    if out_path.exists():
        try:
            existing = {int(k): v for k, v in json.loads(out_path.read_text(encoding="utf-8")).items()}
            print(f"   Loaded {len(existing)} existing results — will skip already-scored entries")
        except Exception as exc:
            print(f"   Could not load existing results ({exc}). Starting fresh.")

    provider = GeminiProvider(model=args.model)
    print(f"Evaluator  : {args.model}")
    print(f"Range      : entries {args.start} to {end_idx}  ({len(entries)} total)")
    print(f"R          : real_world field from {input_path.name}")
    print("-" * 65)

    results: dict[int, dict] = dict(existing)
    c1_scores, c2_scores = [], []

    for local_idx, (scenario, real_world, prompt) in enumerate(entries):
        orig_idx = start_idx + local_idx   # 0-based index into the full entries list
        entry_num = orig_idx + 1           # 1-based for display

        # Skip already-scored entries
        if orig_idx in results:
            prev = results[orig_idx]
            c1 = prev.get("criterion1_score")
            c2 = prev.get("criterion2_score")
            print(f"  entry {entry_num:>3} -- skip (already scored: C1={c1}  C2={c2})")
            if isinstance(c1, int): c1_scores.append(c1)
            if isinstance(c2, int): c2_scores.append(c2)
            continue

        if not prompt.strip():
            print(f"  entry {entry_num:>3} --  empty prompt, skipping")
            continue
        if not real_world.strip():
            print(f"  entry {entry_num:>3} --  empty real_world, skipping")
            continue

        try:
            evaluation = evaluate_prompt(
                provider=provider,
                prompt=prompt,
                real_world=real_world,
            )
        except Exception as exc:
            print(f"  entry {entry_num:>3} -- ERROR: {exc}")
            results[orig_idx] = {
                "entry":          entry_num,
                "scenario":       scenario,
                "prompt_snippet": prompt[:80],
                "error":          str(exc),
            }
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(args.delay)
            continue

        # Attach metadata
        evaluation["entry"]          = entry_num
        evaluation["scenario"]       = scenario
        evaluation["prompt_snippet"] = prompt[:80]

        c1 = evaluation.get("criterion1_score", "?")
        c2 = evaluation.get("criterion2_score", "?")
        if isinstance(c1, int): c1_scores.append(c1)
        if isinstance(c2, int): c2_scores.append(c2)

        print(
            f"  entry {entry_num:>3}   C1(non-thematic)={c1}/5   C2(opacity)={c2}/5"
            f"   | {scenario[:45]}"
        )

        results[orig_idx] = evaluation
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(c1_scores)
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Entries scored   : {n}")
    if n:
        avg1 = sum(c1_scores) / n
        avg2 = sum(c2_scores) / n
        dist1 = dict(sorted((s, c1_scores.count(s)) for s in set(c1_scores)))
        dist2 = dict(sorted((s, c2_scores.count(s)) for s in set(c2_scores)))
        print(f"  C1 non-thematicity   avg: {avg1:.2f}/5   distribution: {dist1}")
        print(f"  C2 opacity           avg: {avg2:.2f}/5   distribution: {dist2}")
    print(f"\n  Full results    -> {out_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_rows = []
    for idx, r in sorted(results.items()):
        summary_rows.append({
            "entry":                r.get("entry", idx + 1),
            "scenario":             r.get("scenario", ""),
            "prompt_snippet":       r.get("prompt_snippet", ""),
            "surface_topic":        r.get("surface_topic", ""),
            "criterion1_score":     r.get("criterion1_score", ""),
            "criterion1_reasoning": r.get("criterion1_reasoning", ""),
            "scenario_recoverable": r.get("scenario_recoverable", ""),
            "criterion2_score":     r.get("criterion2_score", ""),
            "criterion2_reasoning": r.get("criterion2_reasoning", ""),
            "outsider_reconstruction": r.get("outsider_reconstruction", ""),
        })
    summary_path = out_path.with_suffix(".summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  Summary CSV     -> {summary_path}")


if __name__ == "__main__":
    main()
