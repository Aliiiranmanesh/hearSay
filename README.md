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

## 🛠️ Quick Start & Usage

To load and analyze the LLM evaluation scores, you can run the following built-in tools:

```bash
# Calculate aggregate dimension scores across all models
python calculate_averages.py

# Generate a detailed statistical summary report (means, CIs, p-values)
python analyze_scores.py
```

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

