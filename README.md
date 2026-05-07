# HearSayBench: Evaluating Large Language Models on Underrepresented Socio-Legal Scenarios

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Croissant Metadata](https://img.shields.io/badge/Metadata-Croissant_1.0-blue)](https://mlcommons.org/croissant/)
[![NeurIPS Submission](https://img.shields.io/badge/NeurIPS-Datasets_&_Benchmarks_2026-brightgreen)](#)

HearSayBench is a specialized evaluation benchmark designed to assess whether Large Language Models (LLMs) maintain accurate world models for individuals whose real-world struggles are structurally underrepresented in web training corpora.

Applying the **Capabilities Approach** (Sen, 1999; Nussbaum, 2011) as a theoretical framework, HearSayBench models situations where individuals' substantive freedom is restricted by latent, non-demographic barriers (totalitarian controls, travel bans, localized customs, caste systems).

---

## 📂 Repository Structure

```text
HearSayBench/
├── HearSay_Capability_Void_Final.csv  # Labeled 400-scenario dataset (Primary)
├── scenarios_prompts_fixed.csv        # Clean 2-column scenario-prompt pairs (scenarios_prompts.csv)
├── metadata.json                      # Validated Croissant ML metadata file (for OpenReview upload)
├── croissant.json                     # Duplicate of Croissant metadata file

├── entries.txt                        # Source text mapping of all 400 profiles
├── merged/
│   ├── scores.json                    # Performance evaluation scores for 11 LLMs
│   └── harm_scores.json               # Dedicated safety & harm ratings
├── responses/                         # Individual model generation and judgment logs (entry_0001/ to entry_0400/)
├── run_pipeline.py                    # Main test pipeline execution wrapper
├── run_batch.py                       # Batch prompt execution client
├── run_judge.py                       # Automated LLM-as-a-Judge grading client
├── evaluator.py                       # Evaluation metric scoring prompts and definitions
├── harm_eval.py                       # Isolated multi-dimensional safety grader script
├── analyze_scores.py                  # Compiles descriptive performance statistics and 95% CIs
├── calculate_averages.py              # Computes aggregate dimension scores across all models
├── llm_client.py                      # Multi-provider API client wrappers (OpenAI, Gemini, etc.)
└── add_harm_to_merged.py              # Consolidates safety evaluations into the merged database
```


---

## 📜 Dataset Overview

The primary dataset consists of **400 hand-curated, realistic scenarios** mapping implicit first-person user queries to their underlying socio-legal constraints. 

### Key Features
Each record in the primary dataset contains:
1. **Scenario**: The ground-truth real-world context of the underrepresented individual's situation.
2. **Prompt**: The implicit, first-person natural text prompt representing the individual's query (no demographic labels or explicit jargon).
3. **Potential Formal Resources (WEIRD Prior)**: The standard Western-centric recommendations (e.g., calling hotlines, reporting to police) that generic models tend to suggest but are dangerous or ineffective here.
4. **Conversion Factor (The Impediment)**: The specific social, personal, or environmental impediment that negates the resource.
5. **Category**: The broader social constraint domain classification (Social, Personal, Environmental).
6. **Subtype**: The subclass of the social impediment (e.g., Public Policy & Law, Social Norms, Power Relations).

---

## 🛠️ Quick Start & Pipeline Usage

HearSayBench provides a fully modular end-to-end evaluation and grading pipeline. You can run the entire pipeline at once, or execute specific steps (collecting responses, running the evaluator judge, running the harm judge, or compiling statistics) individually.

### 📋 Prerequisites & Installation

1. **Clone and Install Dependencies**:
   ```bash
   pip install google-generativeai openai together python-dotenv pandas numpy scipy
   ```

2. **Configure API Keys**:
   Create a `.env` file in the root of the repository and add your API credentials:
   ```env
   OPENAI_API_KEY="your_openai_key"
   GEMINI_API_KEY="your_gemini_key"
   TOGETHER_API_KEY="your_together_key"
   ANTHROPIC_API_KEY="your_anthropic_key"
   ```

---

### 🚀 Running the Pipeline End-to-End

To run the complete pipeline—from calling LLM APIs for raw responses, grading them with the evaluator and safety judges, merging outputs, and extracting scores—run:

```bash
python run_pipeline.py entries.txt --steps all --model gemini-2.5-flash
```

---

### 🔍 Running Specific Pipeline Steps

If you want to run the pipeline incrementally, you can use the `--steps` argument to invoke specific phases:

#### 1. Gather Model Responses (`responses`)
Reads the profiles in `entries.txt` and calls each active LLM provider (Gemini, OpenAI, Together, Anthropic) to collect their raw advice. Output files are saved in `responses/entry_XXXX/<model_name>/exchange.json`.
```bash
python run_pipeline.py entries.txt --steps responses --delay 1.5
```

#### 2. Grade Performance (`judge`)
Calls the LLM-as-a-Judge using the criteria inside `evaluator.py` to evaluate responses along the four key dimensions (Situational Comprehension, Capability Constraints, Register, and Honesty). Generates `judgment.json` under each entry.
```bash
python run_pipeline.py entries.txt --steps judge --model gemini-2.5-flash
```

#### 3. Evaluate Safety and Harm (`harm`)
Triggers the specialized safety evaluation script (`harm_eval.py`) to rate responses along immediate and structural/societal harm indices. Generates `harm_judgment.json` under each entry.
```bash
python run_pipeline.py entries.txt --steps harm --model gemini-2.5-flash
```

#### 4. Consolidate and Merge Data (`merge,scores`)
Aggregates all raw texts and individual grading judgments, outputs consolidated JSON databases into `merged/`, and extracts performance and safety ratings into `merged/scores.json` and `merged/harm_scores.json`.
```bash
python run_pipeline.py entries.txt --steps merge,scores
```

---

### 📊 Statistical Reporting & Analysis

Once the pipeline has completed, you can compile and analyze the scores using the built-in reporting tools:

#### Calculate Weighted Averages
Computes the final aggregate average score across all dimensions for each model and updates `merged/scores.json`:
```bash
python calculate_averages.py
```

#### Generate Full Statistical Summary
Compiles and prints the detailed paper-grade report containing performance means, 95% confidence intervals, safety indices, and pairwise statistical significance metrics:
```bash
python analyze_scores.py
```


---

### ⚙️ Pipeline Hyperparameters

To ensure standard, rigorous, and reproducible benchmark evaluations, all pipeline components run with explicit generation hyperparameters:

| Pipeline Stage | Script / Task | Default Temperature | Max Completion Tokens | Key Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **Model Advice** | `run_batch.py` (Advice Generation) | `0.7` | `1,024` | Balanced, representative, natural generation behavior. |
| **Performance Judge** | `run_judge.py` (Performance Grading) | `0.2` | `16,384` | Highly deterministic, reproducible grading; high token limit prevents reasoning truncation. |
| **Safety Judge** | `harm_eval.py` (Harm Grading) | `0.1` | `16,384` | Maximum consistency and deterministic risk assessments. |

> [!NOTE]
> Modern reasoning-focused models (e.g., `gpt-5.5` or OpenAI `o`-series models) run with their native internal temperatures and automatically route token budgets via `max_completion_tokens`.


---

## 📄 License & Terms

This dataset and code are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/). 

### Intended Use
- **Evaluation**: Diagnosing safety, bias, and socio-legal situational comprehension in large language models.
- **Academic Research**: Conducting audits on language models regarding global equity, cultural alignment, and international human rights constraints.
- **Non-intended Use**: This dataset is *not* intended to train generative text models on sensitive scenarios or bypass safety guardrails.

---

## 🔧 Maintenance and Update Plan

HearSayBench is maintained by the **HearSayBench Authors**. 
1. **Errata & Retractions**: Any errors in labeling or dataset structure reported via GitHub issues will be addressed quarterly.
2. **Model Evaluations**: As new foundation models are released, they will be benchmarked under the same prompt templates, and the public scores in `merged/` will be updated.
3. **Community Contributions**: We encourage researchers to propose additional socio-legal scenarios from underrepresented regions to expand the benchmark.

---

## 🧬 Responsible AI & Ethics (Croissant Metadata)

We have fully integrated MLCommons Responsible AI (RAI) metadata. The details can be accessed in [metadata.json](file:///c:/Users/alikh/Desktop/HearSay/metadata.json) (or [croissant.json](file:///c:/Users/alikh/Desktop/HearSay/croissant.json)):

- **Data Biases**: Evaluates biases and safety targeting underrepresented populations facing socio-legal or custom-based barriers globally. Geographically focused on documented human rights reports.
- **Data Limitations**: Consists of 400 carefully curated scenarios exclusively in English. Designed as an evaluation diagnostic rather than a training corpus.
- **Personal & Sensitive Information**: **None**. All scenarios, names, and locations are fully anonymized, stylized, and synthetically constructed based on structural human rights templates. Zero PII is present.

