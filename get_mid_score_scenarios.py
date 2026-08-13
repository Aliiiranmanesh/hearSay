#!/usr/bin/env python3
"""
get_mid_score_scenarios.py
==========================
Reads prompt_criteria.json or prompt_criteria.summary.csv and lists the
scenarios/prompts that did not receive scores of 1 or 5 (i.e., scores of 2, 3, or 4)
for Criterion 1 (Non-thematicity) and/or Criterion 2 (Opacity to out-group).

It also attempts to load the full prompt text from entries.txt if present.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

def parse_entries_file(entries_path):
    """
    Parse the entries.txt file format and return a dictionary mapping 
    1-based entry numbers to their full prompt text.
    """
    try:
        if not os.path.exists(entries_path):
            return {}
        
        with open(entries_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n(?=\d+\.\s)", text.strip())
        
        prompts = {}
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Match "N. <scenario>"
            scenario_match = re.match(r"^(\d+)\.\s+(.+)", block)
            if not scenario_match:
                continue
                
            entry_num = int(scenario_match.group(1))
            rest = block.split("\n", 1)[1].strip() if "\n" in block else ""
            
            # Split real-world and prompt by the em-dash line separator
            parts = re.split(r"\n\s*\u2014\s*\n", rest, maxsplit=1)
            if len(parts) >= 2:
                prompts[entry_num] = parts[1].strip()
                
        return prompts
    except Exception as e:
        print(f"Warning: Could not load full prompts from {entries_path}: {e}", file=sys.stderr)
        return {}

def load_from_csv(csv_path):
    """Load entries from the summary CSV file."""
    entries = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entry_num = int(row.get("entry") or 0)
            except ValueError:
                continue
            
            # Helper to safely convert score to int
            def to_score(val):
                try:
                    return int(float(val)) if val else None
                except ValueError:
                    return None
            
            c1_score = to_score(row.get("criterion1_score"))
            c2_score = to_score(row.get("criterion2_score"))
            
            entries.append({
                "entry": entry_num,
                "scenario": row.get("scenario", ""),
                "prompt_snippet": row.get("prompt_snippet", ""),
                "surface_topic": row.get("surface_topic", ""),
                "criterion1_score": c1_score,
                "criterion2_score": c2_score,
                "criterion1_reasoning": row.get("criterion1_reasoning", ""),
                "criterion2_reasoning": row.get("criterion2_reasoning", ""),
                "outsider_reconstruction": row.get("outsider_reconstruction", ""),
                "scenario_recoverable": row.get("scenario_recoverable", ""),
            })
    return entries

def load_from_json(json_path):
    """Load entries from the JSON results file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    entries = []
    # If JSON is structured as a dict of stringified indices: {"0": {...}, "1": {...}}
    if isinstance(data, dict):
        items = data.values()
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported JSON format: top-level element is neither list nor dict")
        
    for item in items:
        # Normalize keys if they differ slightly
        c1_score = item.get("criterion1_score")
        c2_score = item.get("criterion2_score")
        
        entries.append({
            "entry": int(item.get("entry") or 0),
            "scenario": item.get("scenario", ""),
            "prompt_snippet": item.get("prompt_snippet", ""),
            "surface_topic": item.get("surface_topic", ""),
            "criterion1_score": int(c1_score) if c1_score is not None else None,
            "criterion2_score": int(c2_score) if c2_score is not None else None,
            "criterion1_reasoning": item.get("criterion1_reasoning", ""),
            "criterion2_reasoning": item.get("criterion2_reasoning", ""),
            "outsider_reconstruction": item.get("outsider_reconstruction", ""),
            "scenario_recoverable": item.get("scenario_recoverable", ""),
        })
    return entries

def main():
    parser = argparse.ArgumentParser(
        description="Filter and list HearSay scenarios/prompts with scores not equal to 1 or 5 (i.e. 2, 3, or 4)."
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to prompt_criteria.json or prompt_criteria.summary.csv. "
             "If not specified, the script automatically searches for them in the current directory."
    )
    parser.add_argument(
        "--entries", "-e", default="entries.txt",
        help="Path to the entries.txt file to load full prompts (default: entries.txt)."
    )
    parser.add_argument(
        "--criterion", "-c", choices=["c1", "c2", "either", "both"], default="either",
        help="Which criterion to filter on:\n"
             "  c1: Criterion 1 (Non-thematicity) score is 2, 3, or 4\n"
             "  c2: Criterion 2 (Opacity to out-group) score is 2, 3, or 4\n"
             "  either: Either C1 or C2 is 2, 3, or 4 (default)\n"
             "  both: Both C1 and C2 are 2, 3, or 4"
    )
    parser.add_argument(
        "--out", "-o",
        help="Optional path to output the filtered results as a CSV file."
    )
    args = parser.parse_args()

    # Automatically resolve input file if not specified
    input_path = None
    if args.input:
        input_path = Path(args.input)
    else:
        # Search defaults
        for default_name in ["prompt_criteria.summary.csv", "prompt_criteria.json"]:
            p = Path(default_name)
            if p.exists():
                input_path = p
                break
                
    if not input_path or not input_path.exists():
        sys.exit("Error: No input file found. Please specify prompt_criteria.summary.csv or prompt_criteria.json using --input.")

    print(f"Reading evaluations from: {input_path}")
    
    # Load entries
    if input_path.suffix.lower() == ".csv":
        entries = load_from_csv(input_path)
    elif input_path.suffix.lower() == ".json":
        entries = load_from_json(input_path)
    else:
        sys.exit(f"Error: Unknown file format '{input_path.suffix}'. Please provide a .csv or .json file.")
        
    print(f"Loaded {len(entries)} entries.")

    # Load full prompts from entries.txt if available
    full_prompts = parse_entries_file(args.entries)
    if full_prompts:
        print(f"Successfully loaded {len(full_prompts)} full prompts from '{args.entries}' for lookup.")
    else:
        print("Note: entries.txt not found or empty. Using prompt snippets from the results file instead.")

    # Filter logic
    filtered_entries = []
    for entry in entries:
        c1 = entry["criterion1_score"]
        c2 = entry["criterion2_score"]
        
        # Check if score is "mid-range" (not 1 and not 5)
        # We only check scores that are not None and lie in {2, 3, 4}
        is_c1_mid = c1 in [2, 3, 4]
        is_c2_mid = c2 in [2, 3, 4]
        
        match = False
        if args.criterion == "c1":
            match = is_c1_mid
        elif args.criterion == "c2":
            match = is_c2_mid
        elif args.criterion == "either":
            match = is_c1_mid or is_c2_mid
        elif args.criterion == "both":
            match = is_c1_mid and is_c2_mid
            
        if match:
            # Inject full prompt if available
            entry_num = entry["entry"]
            if entry_num in full_prompts:
                entry["prompt"] = full_prompts[entry_num]
            else:
                entry["prompt"] = entry["prompt_snippet"]
            filtered_entries.append(entry)

    # Print summary
    print(f"\nFound {len(filtered_entries)} entries matching filter (Criterion: {args.criterion}):")
    print("=" * 80)
    
    for idx, entry in enumerate(filtered_entries, 1):
        c1_str = str(entry["criterion1_score"]) if entry["criterion1_score"] is not None else "N/A"
        c2_str = str(entry["criterion2_score"]) if entry["criterion2_score"] is not None else "N/A"
        
        print(f"Match #{idx} | Entry {entry['entry']}")
        print(f"Scenario: {entry['scenario']}")
        print(f"Scores  : Criterion 1 (Non-thematicity) = {c1_str}/5, Criterion 2 (Opacity) = {c2_str}/5")
        print(f"Prompt  :")
        # Indent the prompt lines for clear formatting
        prompt_lines = entry["prompt"].splitlines()
        for line in prompt_lines:
            print(f"  {line}")
        print("-" * 80)

    # Save to file if output path is requested
    if args.out:
        out_path = Path(args.out)
        try:
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                # Include 'prompt' field in the CSV row output
                fieldnames = [
                    "entry", "scenario", "criterion1_score", "criterion2_score", 
                    "prompt", "surface_topic", "criterion1_reasoning", 
                    "criterion2_reasoning", "scenario_recoverable", "outsider_reconstruction"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(filtered_entries)
            print(f"\nFiltered results successfully saved to: {out_path}")
        except Exception as e:
            print(f"Error saving to {out_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
