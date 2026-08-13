import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from adjustText import adjust_text
from matplotlib.patches import Patch

def make_accuracy_chart(out_dir):
    # Chart 1 — Deduction accuracy pie chart (7×6 inches)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    
    sizes = [420, 50, 30]  # Updated actual counts: Correct (420), Partial (50), Wrong (30)
    total = sum(sizes)
    colors = ['#1D9E75', '#EF9F27', '#E24B4A']
    explode = (0.02, 0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        colors=colors, 
        autopct='%1.1f%%',
        startangle=90, 
        counterclock=False,
        wedgeprops=dict(edgecolor='white', linewidth=2.5),
        pctdistance=0.70,
        textprops=dict(color='#2C2C2A', fontsize=10, weight='medium')
    )
    
    # Legend labels (percentages only)
    legend_labels = [
        f"Correct ({sizes[0]/total*100:.1f}%)",
        f"Partial ({sizes[1]/total*100:.1f}%)",
        f"Wrong ({sizes[2]/total*100:.1f}%)"
    ]
    
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='white', label=legend_labels[0]),
        Patch(facecolor=colors[1], edgecolor='white', label=legend_labels[1]),
        Patch(facecolor=colors[2], edgecolor='white', label=legend_labels[2])
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        ncol=1,
        fontsize=10,
        labelcolor='#2C2C2A'
    )
    
    plt.title("Deduction accuracy", pad=20, fontsize=14, weight='bold', color='#2C2C2A')
    ax.axis('equal')  
    plt.tight_layout()
    
    p1 = out_dir / "chart_accuracy.png"
    p2 = Path("chart_accuracy.png")
    fig.savefig(p1, dpi=180, facecolor='white', bbox_inches='tight')
    fig.savefig(p2, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy pie chart to {p1} and {p2}")

def make_authenticity_chart(out_dir):
    # Chart 2 — Authenticity assessment bar chart (8×5.5 inches)
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
    
    scores = [1, 2, 3, 4, 5]
    values = [0, 22, 28, 8, 3]
    total = sum(values)
    percentages = [val / total * 100 for val in values]
    colors = ['#E0F2F1', '#80CBC4', '#26A69A', '#00897B', '#004D40']
    
    bars = ax.bar(
        scores, 
        percentages, 
        color=colors, 
        width=0.55, 
        edgecolor='white', 
        linewidth=1.2,
        zorder=3
    )
    
    # Horizontal light gray grid lines behind bars
    ax.grid(axis='y', linestyle='-', color='#E5E5E5', zorder=1)
    ax.set_axisbelow(True)
    
    # Remove visible spines
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Y-axis config
    ax.set_ylim(0, 55)
    ax.set_yticks(range(0, 56, 10))
    ax.set_ylabel("Percentage of prompts (%)", fontsize=11, color='#2C2C2A')
    
    # X-axis two-line labels
    x_labels = [
        "1 /\nAbsolutely uncertain",
        "2 /\nUncertain",
        "3 /\nGeneral",
        "4 /\nFairly confident",
        "5 /\nHighly confident"
    ]
    ax.set_xticks(scores)
    ax.set_xticklabels(x_labels, fontsize=9.5, color='#2C2C2A')
    
    # Labels above each bar (percentages only)
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        if pct == 0.0:
            ax.text(
                bar.get_x() + bar.get_width()/2.0, 
                0.8, 
                "0.0%", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                color='#888780'
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width()/2.0, 
                height + 0.8, 
                f"{pct:.1f}%", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                weight='bold', 
                color='#2C2C2A'
            )
            
    plt.title("Authenticity assessment (Q3)", pad=20, fontsize=13, weight='bold', color='#2C2C2A')
    plt.tight_layout()
    
    p1 = out_dir / "chart_authenticity.png"
    p2 = Path("chart_authenticity.png")
    fig.savefig(p1, dpi=180, facecolor='white', bbox_inches='tight')
    fig.savefig(p2, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved authenticity bar chart to {p1} and {p2}")

def main():
    # Set style to use standard fonts and colors
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'text.color': '#2C2C2A',
        'axes.labelcolor': '#2C2C2A',
        'xtick.color': '#2C2C2A',
        'ytick.color': '#2C2C2A'
    })

    # Load data
    scores_file = Path("merged/scores.json")
    if not scores_file.exists():
        print(f"Error: Could not find scores file at {scores_file}")
        return

    print(f"Reading {scores_file}...")
    with open(scores_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # All 6 core dimensions we want to track
    dims = [
        "situational_comprehension",
        "capability_and_substantive_freedom",
        "register_appropriateness",
        "honesty_about_uncertainty",
        "immediate_harm",
        "societal_harm"
    ]

    # Accumulate scores per model
    model_data = {}
    for prompt, models in data.items():
        if not isinstance(models, dict):
            continue
        for model_name, metrics in models.items():
            if not isinstance(metrics, dict):
                continue
            
            if model_name not in model_data:
                model_data[model_name] = {d: [] for d in dims}
                model_data[model_name]["average"] = []
                model_data[model_name]["harm_average"] = []
            
            # Extract each dimension if present
            for d in dims:
                val = metrics.get(d)
                if val is not None and isinstance(val, (int, float)):
                    model_data[model_name][d].append(val)
            
            # Also extract the precomputed averages
            avg_score = metrics.get("average")
            harm_avg = metrics.get("harm_average")
            
            if avg_score is not None:
                model_data[model_name]["average"].append(avg_score)
            if harm_avg is not None:
                model_data[model_name]["harm_average"].append(harm_avg)

    # Compute averages
    records = []
    for model_name, lists in model_data.items():
        n = len(lists["situational_comprehension"])
        if n > 0:
            row = {"Model": model_name, "N": n}
            # Add mean for each dimension
            for d in dims:
                lst = lists[d]
                row[d] = round(sum(lst) / len(lst), 3) if lst else 0.0
                
            # Add aggregate average scores
            avg_lst = lists["average"]
            harm_lst = lists["harm_average"]
            row["Weighted Capability & Substantive Freedom Score"] = round(sum(avg_lst) / len(avg_lst), 3) if avg_lst else 0.0
            row["Safety (Harm Average)"] = round(sum(harm_lst) / len(harm_lst), 3) if harm_lst else 0.0
            
            records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No valid model data found.")
        return

    # Print summary table
    print("\nModel Performance Summary:")
    print(df.to_string(index=False))

    # Output directory
    out_dir = Path("merged/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort models by overall capability score for consistent presentation
    df_ranked_cap = df.sort_values(by="Weighted Capability & Substantive Freedom Score", ascending=False)
    model_order = df_ranked_cap["Model"].tolist()

    # ----------------------------------------------------
    # Chart 1: Ranked Capability Bar Chart
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(df_ranked_cap))
    ax = sns.barplot(
        x="Weighted Capability & Substantive Freedom Score", 
        y="Model", 
        data=df_ranked_cap, 
        palette=colors,
        hue="Model",
        legend=False
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=5)

    plt.title("HearSayBench: Model Capability & Substantive Freedom (Ranked)", pad=15)
    plt.xlabel("Weighted Capability & Substantive Freedom Score (1 to 5, higher is better)")
    plt.ylabel("Model")
    plt.xlim(1, 5)
    plt.tight_layout()
    chart1_path = out_dir / "model_capability_ranking.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved Chart 1 to: {chart1_path.resolve()}")

    # ----------------------------------------------------
    # Chart 2: Capability vs. Safety Trade-off Scatter Plot
    # ----------------------------------------------------
    plt.figure(figsize=(11, 9))
    scatter = sns.scatterplot(
        x="Weighted Capability & Substantive Freedom Score",
        y="Safety (Harm Average)",
        data=df,
        s=160,
        color="royalblue",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8,
        zorder=3
    )

    texts = []
    for idx, row in df.iterrows():
        t = plt.text(
            row["Weighted Capability & Substantive Freedom Score"], 
            row["Safety (Harm Average)"], 
            row["Model"], 
            fontsize=9.5,
            weight='medium',
            zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        )
        texts.append(t)

    adjust_text(
        texts,
        x=df["Weighted Capability & Substantive Freedom Score"].tolist(),
        y=df["Safety (Harm Average)"].tolist(),
        arrowprops=dict(arrowstyle="->", color='dimgray', lw=0.8, alpha=0.7),
        expand=(1.25, 1.35),
        force_text=(0.3, 0.4),
        zorder=4
    )

    plt.title("HearSayBench: Capability & Substantive Freedom vs. Safety (Harm)", pad=15)
    plt.xlabel("Weighted Capability & Substantive Freedom Score (1-5, higher is better)")
    plt.ylabel("Safety (Harm Average, 1-5, higher is safer / less harm)")
    plt.xlim(1.5, 4.5)
    plt.ylim(1.5, 3.5)
    plt.axvline(x=df["Weighted Capability & Substantive Freedom Score"].mean(), color='red', linestyle='--', alpha=0.5, label="Mean Capability")
    plt.axhline(y=df["Safety (Harm Average)"].mean(), color='green', linestyle='--', alpha=0.5, label="Mean Safety")
    plt.legend(loc="upper left")
    plt.tight_layout()
    chart2_path = out_dir / "capability_vs_safety_tradeoff.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved Chart 2 to: {chart2_path.resolve()}")
    
    # ----------------------------------------------------
    # Chart 3: Combined Rank Comparison (Double Bar Chart)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 8))
    df_melted = df.melt(id_vars=["Model"], value_vars=["Weighted Capability & Substantive Freedom Score", "Safety (Harm Average)"],
                        var_name="Metric", value_name="Score")
    
    sns.barplot(
        x="Score",
        y="Model",
        hue="Metric",
        data=df_melted,
        order=model_order,
        palette="muted"
    )
    plt.title("HearSayBench: Model Capability vs. Safety Comparison", pad=15)
    plt.xlabel("Score (1 to 5, higher is better)")
    plt.ylabel("Model")
    plt.xlim(1, 5)
    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    chart3_path = out_dir / "capability_vs_safety_bars.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved Chart 3 to: {chart3_path.resolve()}")

    # ----------------------------------------------------
    # Chart 4: Model Performance Heatmap (All 6 Dimensions)
    # ----------------------------------------------------
    plt.figure(figsize=(13, 9))
    
    # Human-readable mapping for axes
    heatmap_dims = {
        "situational_comprehension": "Situational Comp.",
        "capability_and_substantive_freedom": "Capability / Freedom",
        "register_appropriateness": "Register Appropriateness",
        "honesty_about_uncertainty": "Honesty / Uncertainty",
        "immediate_harm": "Immediate Harm",
        "societal_harm": "Societal Harm"
    }
    
    # Re-order and rename columns
    heatmap_data = df_ranked_cap.set_index("Model")[list(heatmap_dims.keys())]
    heatmap_data = heatmap_data.rename(columns=heatmap_dims)
    
    # Create the heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=1.0,
        vmax=5.0,
        linewidths=.6,
        cbar_kws={'label': 'Score (1 to 5 Scale, Higher is Better / Safer)'}
    )
    
    plt.title("HearSayBench: Detailed Performance across All 6 Core Dimensions", pad=20)
    plt.ylabel("Model (Sorted by Capability Score)")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    chart4_path = out_dir / "model_dimensions_heatmap.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"Saved Chart 4 to: {chart4_path.resolve()}")

    # ----------------------------------------------------
    # Chart 5: Dimensional Stacked/Grouped Bar Chart
    # ----------------------------------------------------
    plt.figure(figsize=(14, 10))
    
    # Reshape for multi-group bar chart
    df_dims_melted = df.melt(
        id_vars=["Model"], 
        value_vars=dims,
        var_name="Dimension", 
        value_name="Score"
    )
    df_dims_melted["Dimension"] = df_dims_melted["Dimension"].map(heatmap_dims)
    
    sns.barplot(
        x="Score",
        y="Model",
        hue="Dimension",
        data=df_dims_melted,
        order=model_order,
        palette="husl"
    )
    plt.title("HearSayBench: Model Scores on Individual Capability & Safety Dimensions", pad=15)
    plt.xlabel("Score (1 to 5)")
    plt.ylabel("Model")
    plt.xlim(1, 5)
    plt.legend(title="Dimension", loc="lower right", bbox_to_anchor=(1, 0))
    plt.tight_layout()
    chart5_path = out_dir / "model_dimensions_bars.png"
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"Saved Chart 5 to: {chart5_path.resolve()}")

    # ----------------------------------------------------
    # Publication-ready requested charts (Chart 6 and 7)
    # ----------------------------------------------------
    make_accuracy_chart(out_dir)
    make_authenticity_chart(out_dir)

    print("\nSuccessfully generated all charts in merged/charts/")

if __name__ == "__main__":
    main()
