import json
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

def mean(data):
    return sum(data) / len(data) if data else 0.0

def variance(data, mu=None):
    if len(data) <= 1:
        return 0.0
    if mu is None:
        mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / (len(data) - 1)

def std_dev(data, mu=None):
    return math.sqrt(variance(data, mu))

def median(data):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

def pearson_r(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = mean(x)
    mean_y = mean(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y))
    return num / den if den != 0 else 0.0

def mean_abs_diff(x, y):
    return sum(abs(xi - yi) for xi, yi in zip(x, y)) / len(x) if x else 0.0

def cohen_d(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = mean(x)
    mean_y = mean(y)
    var_x = variance(x, mean_x)
    var_y = variance(y, mean_y)
    pooled_std = math.sqrt((var_x + var_y) / 2.0)
    return (mean_x - mean_y) / pooled_std if pooled_std != 0 else 0.0

def cohen_kappa(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    
    agreements = sum(1 for xi, yi in zip(x, y) if xi == yi)
    po = agreements / n
    
    counts_x = Counter(x)
    counts_y = Counter(y)
    
    pe = 0.0
    for category in set(x) | set(y):
        prob_x = counts_x.get(category, 0) / n
        prob_y = counts_y.get(category, 0) / n
        pe += prob_x * prob_y
        
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)

def make_pair_header(m1, m2):
    s1 = m1[:4].ljust(4)
    s2 = m2[:4].ljust(4)
    return f"{s1}x    {s2}"

def main():
    parser = argparse.ArgumentParser(description="Generate a comprehensive analysis report on benchmark scores.")
    parser.add_argument("--input", default="merged/scores.json", help="Path to scores JSON (default: merged/scores.json)")
    parser.add_argument("--out", default="merged/analyze_scores_report.txt", help="Output report path (default: merged/analyze_scores_report.txt)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Scores file not found at '{input_path}'")
        return

    try:
        scores_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return

    # Discover unique active models
    raw_models = set()
    for prompt, providers in scores_data.items():
        if isinstance(providers, dict):
            raw_models.update(providers.keys())
    raw_models = sorted(list(raw_models))

    if not raw_models:
        print("Error: No models found in scores file.")
        return

    # Dimension definitions
    judgment_dimensions = [
        "situational_comprehension",
        "capability_and_substantive_freedom",
        "constraint_respect",
        "register_appropriateness",
        "honesty_about_uncertainty"
    ]

    dimension_labels = {
        "situational_comprehension": "Situational Comp.",
        "capability_and_substantive_freedom": "Capability/Freedom",
        "constraint_respect": "Capability/Freedom",
        "register_appropriateness": "Register Approp.",
        "honesty_about_uncertainty": "Honesty/Uncertainty"
    }

    # Discover active dimensions in this dataset
    active_dims = set()
    for prompt, providers in scores_data.items():
        if not isinstance(providers, dict):
            continue
        for p, metrics in providers.items():
            if not isinstance(metrics, dict):
                continue
            for k in metrics.keys():
                if k in judgment_dimensions:
                    active_dims.add(k)
    active_dims = [d for d in judgment_dimensions if d in active_dims]

    # Dynamically determine the set of active models (i.e. those with complete metrics for at least one entry)
    all_models = []
    for m in raw_models:
        complete_count = 0
        for prompt, providers in scores_data.items():
            if isinstance(providers, dict) and m in providers and isinstance(providers[m], dict):
                metrics = providers[m]
                if all(metrics.get(d) is not None and isinstance(metrics.get(d), (int, float)) for d in active_dims):
                    complete_count += 1
        if complete_count > 0:
            all_models.append(m)
            
    all_models = sorted(all_models)

    if not all_models:
        print("Error: No models with complete score dimensions found.")
        return

    # Determine global fully-scored prompts where ALL active models have all dimensions
    fully_scored_prompts = []
    for prompt, providers in scores_data.items():
        if not isinstance(providers, dict):
            continue
        if all(m in providers and isinstance(providers[m], dict) and all(providers[m].get(d) is not None and isinstance(providers[m].get(d), (int, float)) for d in active_dims) for m in all_models):
            fully_scored_prompts.append(prompt)

    total_prompts = len(scores_data)
    n_fully_scored = len(fully_scored_prompts)

    # Gather data lists for metrics computation on ALL prompts where the specific model is fully scored
    # (Per-Model stats are calculated per-model on its own N)
    model_averages = {m: [] for m in all_models}
    model_dim_scores = {m: {d: [] for d in active_dims} for m in all_models}
    model_harm_scores = {m: {"immediate_harm": [], "societal_harm": [], "harm_average": []} for m in all_models}

    for prompt, providers in scores_data.items():
        if not isinstance(providers, dict):
            continue
        for m in all_models:
            if m in providers and isinstance(providers[m], dict):
                metrics = providers[m]
                # Check if this model is fully scored on this prompt
                if all(metrics.get(d) is not None and isinstance(metrics.get(d), (int, float)) for d in active_dims):
                    
                    # Fetch or calculate average on the fly
                    avg_val = metrics.get("average")
                    if avg_val is None:
                        dim_vals = [metrics[d] for d in active_dims]
                        avg_val = mean(dim_vals) if dim_vals else 0.0
                    
                    model_averages[m].append(avg_val)
                    
                    for dim in active_dims:
                        model_dim_scores[m][dim].append(metrics[dim])
                        
                    # Also collect harm scores if available
                    imm = metrics.get("immediate_harm")
                    soc = metrics.get("societal_harm")
                    harm_avg = metrics.get("harm_average")
                    if harm_avg is None and imm is not None and soc is not None:
                        harm_avg = (imm + soc) / 2.0
                        
                    if imm is not None and isinstance(imm, (int, float)):
                        model_harm_scores[m]["immediate_harm"].append(imm)
                    if soc is not None and isinstance(soc, (int, float)):
                        model_harm_scores[m]["societal_harm"].append(soc)
                    if harm_avg is not None and isinstance(harm_avg, (int, float)):
                        model_harm_scores[m]["harm_average"].append(harm_avg)

    # Compute Statistics for each model
    stats = {}
    for m in all_models:
        averages = model_averages[m]
        mu = mean(averages)
        sd = std_dev(averages, mu)
        var = variance(averages, mu)
        n_model = len(averages)
        
        # 95% Confidence Interval
        ci_margin = 1.960 * (sd / math.sqrt(n_model)) if n_model > 0 else 0.0
        ci_lower = mu - ci_margin
        ci_upper = mu + ci_margin
        
        stats[m] = {
            "n": float(n_model),
            "mean": mu,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "median": median(averages),
            "std": sd,
            "var": var,
            "min": float(min(averages)) if averages else 0.0,
            "max": float(max(averages)) if averages else 0.0,
            "q1": percentile(averages, 0.25),
            "q3": percentile(averages, 0.75),
        }

    # Format the report
    lines = []
    lines.append("=" * 80)
    lines.append("  HearSayBench Score Analysis")
    lines.append("=" * 80)
    lines.append(f"  File              : {input_path.resolve()}")
    lines.append(f"  Total prompts     : {total_prompts}")
    lines.append(f"  Models            : {', '.join(all_models)}")
    lines.append(f"  Dimensions        : {', '.join(active_dims)}")
    lines.append(f"  Fully-scored      : {n_fully_scored} prompts (all models x all dims)")
    lines.append("")

    # Align columns dynamically based on 12-character width
    col_width = 12
    header_row = f"{'':<29}" + "".join(f"{m:>{col_width}}" for m in all_models)
    separator = "  " + "-" * (29 + col_width * len(all_models))

    lines.append(f"-- Per-Model Stats - Aggregate Score (mean of 4 dimensions) ".ljust(76, "-"))
    lines.append(header_row)
    lines.append(separator)

    stat_labels = [
        ("N (fully scored)", "n", ".3f"),
        ("Mean", "mean", ".3f"),
        ("95% CI lower", "ci_lower", ".3f"),
        ("95% CI upper", "ci_upper", ".3f"),
        ("Median", "median", ".3f"),
        ("Std Dev", "std", ".3f"),
        ("Variance", "var", ".3f"),
        ("Min", "min", ".3f"),
        ("Max", "max", ".3f"),
        ("Q1 (25%)", "q1", ".3f"),
        ("Q3 (75%)", "q3", ".3f")
    ]

    for label, key, fmt in stat_labels:
        row = f"  {label:<29}"
        for m in all_models:
            val = stats[m][key]
            row += f"{val:>{col_width}{fmt}}"
        lines.append(row)
    lines.append("")

    # Per-Dimension Mean Scores
    lines.append(f"-- Per-Dimension Mean Scores (by model) ".ljust(76, "-"))
    lines.append(header_row)
    lines.append(separator)
    for dim in active_dims:
        label = dimension_labels.get(dim, dim)
        row = f"  {label:<29}"
        for m in all_models:
            d_mean = mean(model_dim_scores[m][dim])
            row += f"{d_mean:>{col_width}.3f}"
        lines.append(row)
    lines.append("")

    # Harm Evaluation Mean Scores
    lines.append(f"-- Harm Evaluation Mean Scores (by model) ".ljust(76, "-"))
    lines.append(header_row)
    lines.append(separator)
    harm_labels = [
        ("Immediate Harm (A)", "immediate_harm"),
        ("Societal Harm (B)", "societal_harm"),
        ("Harm Average", "harm_average")
    ]
    for label, key in harm_labels:
        row = f"  {label:<29}"
        for m in all_models:
            h_mean = mean(model_harm_scores[m][key])
            row += f"{h_mean:>{col_width}.3f}"
        lines.append(row)
    lines.append("")

    # Aggregate Score Distribution
    lines.append(f"-- Aggregate Score Distribution (rounded to nearest int) ".ljust(76, "-"))
    dist_header = f"   {'Score':<8}" + "".join(f"{m:>{col_width}}" for m in all_models)
    lines.append(dist_header)
    lines.append("  " + "-" * (8 + col_width * len(all_models)))
    
    for score_int in range(1, 6):
        row = f"     {score_int:<8}"
        for m in all_models:
            rounded_vals = [round(val) for val in model_averages[m]]
            cnt = sum(1 for v in rounded_vals if v == score_int)
            total_cnt = len(rounded_vals)
            pct = round(cnt * 100 / total_cnt) if total_cnt > 0 else 0
            pct_str = f"({pct:>4}%)"
            formatted = f"{cnt:>4}{pct_str:<6}"
            row += f"{formatted:>{col_width}}"
        lines.append(row)
    lines.append("")

    # Model Ranking by Aggregate
    lines.append(f"-- Model Ranking by Aggregate Mean Score ".ljust(76, "-"))
    lines.append(f"  {'Rank':<6}{'Model':<23}{'Mean':>10}{'Std':>10}{'95% CI':^22}")
    lines.append("  " + "-" * 73)
    
    sorted_ranking = sorted(all_models, key=lambda m: stats[m]["mean"], reverse=True)
    for idx, m in enumerate(sorted_ranking, start=1):
        mean_val = stats[m]["mean"]
        std_val = stats[m]["std"]
        ci_str = f"[{stats[m]['ci_lower']:.3f}, {stats[m]['ci_upper']:.3f}]"
        lines.append(f"  #{idx:<4}{m:<23}{mean_val:>10.3f}{std_val:>10.3f}  {ci_str:^20}")
    lines.append("")

    # Model Ranking by Harm Average
    lines.append(f"-- Model Ranking by Harm Average (Safety - Higher is Better) ".ljust(76, "-"))
    lines.append(f"  {'Rank':<6}{'Model':<23}{'Mean Harm':>12}{'Std Harm':>12}{'N Evaluated':>14}")
    lines.append("  " + "-" * 71)
    
    sorted_harm_ranking = sorted(all_models, key=lambda m: mean(model_harm_scores[m]["harm_average"]), reverse=True)
    for idx, m in enumerate(sorted_harm_ranking, start=1):
        mean_harm = mean(model_harm_scores[m]["harm_average"])
        std_harm = std_dev(model_harm_scores[m]["harm_average"])
        n_eval = len(model_harm_scores[m]["harm_average"])
        lines.append(f"  #{idx:<4}{m:<23}{mean_harm:>12.3f}{std_harm:>12.3f}{n_eval:>14}")
    lines.append("")

    # Pairwise Agreement Section (on global fully-scored subset)
    lines.append(f"-- Pairwise Agreement (on {n_fully_scored} fully-scored prompts, aggregate scores) ".ljust(76, "-"))
    
    # Pre-build list of global averages for calculations to ensure perfect alignment
    global_model_averages = {m: [] for m in all_models}
    global_model_dim_scores = {m: {d: [] for d in active_dims} for m in all_models}
    
    for idx_prompt, prompt in enumerate(fully_scored_prompts):
        providers = scores_data[prompt]
        for m in all_models:
            metrics = providers[m]
            
            avg_val = metrics.get("average")
            if avg_val is None:
                dim_vals = [metrics[d] for d in active_dims]
                avg_val = mean(dim_vals) if dim_vals else 0.0
            
            global_model_averages[m].append(avg_val)
            for dim in active_dims:
                global_model_dim_scores[m][dim].append(metrics[dim])

    # Compute Exact and Within-1 agreement on global averages
    exact_count = 0
    within_1_count = 0
    for idx_prompt in range(n_fully_scored):
        prompt_avgs = [global_model_averages[m][idx_prompt] for m in all_models]
        if len(set(prompt_avgs)) == 1:
            exact_count += 1
        if max(prompt_avgs) - min(prompt_avgs) <= 1.0:
            within_1_count += 1
            
    pct_exact = (exact_count * 100 / n_fully_scored) if n_fully_scored > 0 else 0.0
    pct_within_1 = (within_1_count * 100 / n_fully_scored) if n_fully_scored > 0 else 0.0
    
    lines.append(f"  Exact agreement (all models same): {exact_count}/{n_fully_scored} ({pct_exact:.1f}%)")
    lines.append(f"  Within-1 agreement (spread <= 1) : {within_1_count}/{n_fully_scored} ({pct_within_1:.1f}%)")
    lines.append("")

    # Pairwise Table Header
    lines.append(f"  {'Pair':<36}{'Kappa':>10}{'Pearson r':>12}{'Mean |diff|':>12}{'Cohen d':>10}")
    lines.append("  " + "-" * 82)

    # Compute and display pairwise agreements
    pairs_done = set()
    pairs_list = []
    for m1 in all_models:
        for m2 in all_models:
            if m1 == m2 or (m2, m1) in pairs_done:
                continue
            pairs_done.add((m1, m2))
            pairs_list.append((m1, m2))
            
            x = global_model_averages[m1]
            y = global_model_averages[m2]
            
            # Kappa on rounded integer values
            x_round = [round(val) for val in x]
            y_round = [round(val) for val in y]
            kappa_val = cohen_kappa(x_round, y_round)
            
            r_val = pearson_r(x, y)
            diff_val = mean_abs_diff(x, y)
            d_val = cohen_d(x, y)
            
            pair_name = f"{m1} vs {m2}"
            lines.append(f"  {pair_name:<36}{kappa_val:>10.3f}{r_val:>12.3f}{diff_val:>12.3f}{d_val:>10.3f}")
    lines.append("")

    # Per-Dimension Pairwise Pearson r
    lines.append(f"-- Per-Dimension Pairwise Pearson r (model correlation) ".ljust(76, "-"))
    
    # 29 label spaces, then consecutive 13-character columns
    col_width_p = 13
    pearson_header = f"  {'Dimension':<27}" + "".join(make_pair_header(m1, m2) for m1, m2 in pairs_list)
    lines.append(pearson_header)
    lines.append("  " + "-" * (27 + col_width_p * len(pairs_list)))
    
    for dim in active_dims:
        label = dimension_labels.get(dim, dim)
        row = f"  {label:<27}"
        for m1, m2 in pairs_list:
            x_dim = global_model_dim_scores[m1][dim]
            y_dim = global_model_dim_scores[m2][dim]
            r_dim = pearson_r(x_dim, y_dim)
            row += f"{r_dim:>{col_width_p}.3f}"
        lines.append(row)

    report_text = "\n".join(lines) + "\n"

    # Save to file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"SUCCESS: Analysis report generated and saved to: {out_path}\n")

    # Output to stdout safely by encoding as utf-8 or replacing unsupported chars
    # We can write directly to sys.stdout.buffer
    sys.stdout.buffer.write(report_text.encode('utf-8'))

if __name__ == "__main__":
    main()
