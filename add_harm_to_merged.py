import json
from pathlib import Path
import argparse

def add_harm_to_merged(responses_dir: Path, merged_dir: Path):
    """
    1. Updates existing merged/<model>.json files with harm_evaluation data.
    2. Updates scores.json with numerical harm scores (immediate_harm, societal_harm, harm_average).
    3. Creates merged/harm_scores.json with numerical harm scores.
    """
    responses_dir = Path(responses_dir)
    merged_dir = Path(merged_dir)
    
    # Dynamically find all unique model folders under each entry_* directory
    entry_dirs = sorted(responses_dir.glob("entry_*"))
    models = sorted(list({
        sub.name
        for entry_dir in entry_dirs
        for sub in entry_dir.iterdir()
        if sub.is_dir()
    }))
    
    print(f"Discovered {len(models)} models/providers: {', '.join(models)}")
    
    # 1. Update model JSON files in merged/
    print("\nUpdating merged model files...")
    for model in models:
        merged_path = merged_dir / f"{model}.json"
        if not merged_path.exists():
            print(f"  Skipping {model}: {merged_path} not found.")
            continue
            
        merged_data = json.loads(merged_path.read_text(encoding="utf-8"))
        updated_count = 0
        
        # Iterate through entry directories to find harm_judgment.json
        for entry_dir in entry_dirs:
            harm_path = entry_dir / model / "harm_judgment.json"
            if not harm_path.exists():
                continue
                
            try:
                harm_data = json.loads(harm_path.read_text(encoding="utf-8"))
                prompt = harm_data.get("prompt")
                
                if prompt in merged_data:
                    merged_data[prompt]["harm_evaluation"] = harm_data["harm_evaluation"]
                    updated_count += 1
            except Exception as e:
                print(f"  Error reading {harm_path}: {e}")
        
        merged_path.write_text(json.dumps(merged_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Updated {merged_path.name} with {updated_count} harm evaluations.")

    # Build prompt -> entry_dir mapping upfront to make the search extremely fast (O(N))
    print("\nMapping prompts to entry directories...")
    prompt_to_entry = {}
    for entry_dir in entry_dirs:
        for sub in entry_dir.iterdir():
            if sub.is_dir():
                ex_path = sub / "exchange.json"
                if ex_path.exists():
                    try:
                        ex = json.loads(ex_path.read_text(encoding="utf-8"))
                        prompt = ex.get("prompt") or next(iter(ex.keys()))
                        if prompt:
                            prompt_to_entry[prompt] = entry_dir
                            break
                    except:
                        pass

    # 2. Update scores.json
    print("\nUpdating scores.json...")
    scores_path = merged_dir / "scores.json"
    if scores_path.exists():
        scores_data = json.loads(scores_path.read_text(encoding="utf-8"))
        scores_updated = 0
        
        for prompt, models_data in scores_data.items():
            entry_dir = prompt_to_entry.get(prompt)
            if not entry_dir:
                continue
            
            for model in models:
                harm_path = entry_dir / model / "harm_judgment.json"
                if harm_path.exists():
                    try:
                        harm_data = json.loads(harm_path.read_text(encoding="utf-8"))
                        eval_data = harm_data.get("harm_evaluation", {})
                        
                        imm = eval_data.get("immediate_harm", {}).get("score")
                        soc = eval_data.get("societal_harm", {}).get("score")
                        
                        if models_data.get(model) is None:
                            models_data[model] = {}
                            
                        models_data[model]["immediate_harm"] = imm
                        models_data[model]["societal_harm"] = soc
                        
                        if imm is not None and soc is not None:
                            models_data[model]["harm_average"] = round((imm + soc) / 2, 2)
                        
                        scores_updated += 1
                    except Exception as e:
                        pass
                    
        scores_path.write_text(json.dumps(scores_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Updated {scores_path.name} with {scores_updated} harm score entries.")

    # 3. Create harm_scores.json
    print("\nCreating harm_scores.json...")
    harm_scores = {}
    
    for entry_dir in entry_dirs:
        prompt = None
        # Try to find the prompt from any available harm_judgment.json in this entry
        for model in models:
            harm_path = entry_dir / model / "harm_judgment.json"
            if harm_path.exists():
                try:
                    harm_data = json.loads(harm_path.read_text(encoding="utf-8"))
                    prompt = harm_data.get("prompt")
                    if prompt:
                        break
                except:
                    continue
        
        # If still no prompt, check exchange.json
        if not prompt:
            for model in models:
                ex_path = entry_dir / model / "exchange.json"
                if ex_path.exists():
                    try:
                        ex = json.loads(ex_path.read_text(encoding="utf-8"))
                        prompt = ex.get("prompt") or next(iter(ex.keys()))
                        if prompt:
                            break
                    except:
                        continue

        if not prompt:
            continue

        entry_harm_scores = {}
        for model in models:
            harm_path = entry_dir / model / "harm_judgment.json"
            if harm_path.exists():
                try:
                    harm_data = json.loads(harm_path.read_text(encoding="utf-8"))
                    eval_data = harm_data.get("harm_evaluation", {})
                    
                    imm = eval_data.get("immediate_harm", {}).get("score")
                    soc = eval_data.get("societal_harm", {}).get("score")
                    
                    entry_harm_scores[model] = {
                        "immediate_harm": imm,
                        "societal_harm": soc
                    }
                    if imm is not None and soc is not None:
                        entry_harm_scores[model]["average"] = round((imm + soc) / 2, 2)
                except:
                    entry_harm_scores[model] = None
            else:
                entry_harm_scores[model] = None
        
        harm_scores[prompt] = entry_harm_scores

    harm_scores_path = merged_dir / "harm_scores.json"
    harm_scores_path.write_text(json.dumps(harm_scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Created {harm_scores_path.name} with {len(harm_scores)} entries.")

def main():
    parser = argparse.ArgumentParser(description="Add harm evaluations to merged files.")
    parser.add_argument("--responses", default="responses", help="Responses directory")
    parser.add_argument("--merged", default="merged", help="Merged directory")
    args = parser.parse_args()
    
    add_harm_to_merged(Path(args.responses), Path(args.merged))

if __name__ == "__main__":
    main()
