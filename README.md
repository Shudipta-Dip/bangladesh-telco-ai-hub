# Bangladesh Telco AI Hub: Unit Economics Simulator

A simulation engine that models subscriber usage, wholesale API bills, and profit margins for retail AI subscriptions launched by Bangladesh mobile network operators: Banglalink, Robi, and Grameenphone.

---

## Overview

In late 2026, Bangladesh's three major mobile operators launched retail consumer AI plans. Each operator picked a different commercial approach:

1. **Banglalink (RYZE):** Sells flat monthly subscriptions with model whitelists, fixed media quotas, and file uploads. Digital checkouts absorb 15% VAT inside the sticker fee.
2. **Robi (EveryAI):** Sells prepaid monthly token pools with mode routing. Every generated image or file deducts tokens directly from the user balance.
3. **Grameenphone (OneAI):** Sells monthly subscriptions with model whitelists, adding 15% VAT on top of checkout fees.

This simulator tests whether these packaging strategies produce sustainable margins or expose operators to runaway wholesale API bills.

---

## Key Findings

### 1. Banglalink Creator Plan (৳1,000) Loses Money
Banglalink bundles unlimited chat, 272 images, 65 video clips, 389 audio tracks, and 30 file uploads into a flat ৳1,000 monthly fee. When active subscribers use their media quotas, wholesale provider bills exceed subscription revenue. In realistic cohort simulations, Creator averages a **-20.4% gross margin**, with heavy users generating losses up to ৳5,000 per subscriber.

### 2. Robi Token Pool Deductions Stop Runaway Costs
Robi avoids runaway media bills by converting image and file generation into token deductions from each subscriber prepaid balance. Because the deduction rate matches retail prices, wholesale media bills cannot exceed collected revenue. All five Robi monthly plans maintain healthy profit margins between **47.0% and 67.4%**.

### 3. Grameenphone Protects Margins Through Tax Policy and Model Tiers
Grameenphone collects 15% VAT on top of catalog prices, keeping 100% of the sticker price as recognized revenue. GP also locks expensive models (like Claude Opus and GPT-5.6) behind its ৳799 Pro tier, while keeping lower tiers on efficient models like Grok 4.1 Fast and Gemini 3 Flash. All four paid GP tiers remain profitable, with margins between **23.4% and 51.5%**.

---

## Plan Comparison

| Operator | Plan | Price (BDT) | Allowance Type | Media Quota | Modeled Margin |
|---|---|---:|---|---|---:|
| **Banglalink** | Basic | ৳100 | Unlimited Chat (3 models) | 5 images, 0 video, 0 audio | 65.2% |
| **Banglalink** | Advanced | ৳300 | Unlimited Chat (4 models) | 15 images, 0 video, 0 audio | 85.2% |
| **Banglalink** | Pro | ৳500 | Unlimited Chat (5 models) | 54 images, 13 video clips | 56.4% |
| **Banglalink** | Creator | ৳1,000 | Unlimited Chat (11 models) | 272 images, 65 clips, 389 audio | -20.4% |
| **Banglalink** | Legend | ৳1,500 | Unlimited Chat (8 models) | 272 images, 65 clips, 389 audio | 10.1% |
| **Robi** | Tk 99 | ৳99 | 1,000,000 Token Pool | Deducts from pool | 67.4% |
| **Robi** | Tk 149 | ৳149 | 2,000,000 Token Pool | Deducts from pool | 63.6% |
| **Robi** | Tk 349 | ৳349 | 5,000,000 Token Pool | Deducts from pool | 56.9% |
| **Robi** | Tk 599 | ৳599 | 10,000,000 Token Pool | Deducts from pool | 52.3% |
| **Robi** | Tk 999 | ৳999 | 20,000,000 Token Pool | Deducts from pool | 47.0% |
| **Grameenphone** | Student | ৳71 | Unlimited Chat (2 models) | 10 images, 0 video | 39.8% |
| **Grameenphone** | Lite | ৳99 | Unlimited Chat (4 models) | 30 images, 0 video | 23.4% |
| **Grameenphone** | Plus | ৳299 | Unlimited Chat (8 models) | 50 images, 0 video | 51.5% |
| **Grameenphone** | Pro | ৳799 | Unlimited Chat (6 frontier) | 100 images, 30s video | 46.9% |

---

## How It Works

The simulation engine evaluates unit economics through three complementary methods:

1. **Exhaustive Factor Grid:** Explores every permutation of model choices, prompt-to-completion ratios, cache hit rates, and media utilization to establish theoretical profit and loss boundaries.
2. **Empirical Monte Carlo:** Models 10,000 cohorts of 100 users (1,000,000 total subscribers) using realistic behavior distributions. Accounts for user segments (dormant, core, power), prompt caching rates, input-to-output ratios, and variable media lengths.
3. **Scenario Stress Tests:** Re-runs identical user cohorts across 12 stress scenarios, including foreign exchange swings, regulatory tax shifts, wholesale provider price increases, and server routing overheads.

---

## Repository Structure

```text
├── bd_info.md          # Market research, plan quotas, and provider price cards
├── simulation_spec.md  # Core accounting formulas and user segmentation spec
├── requirements.txt    # Python dependencies
├── sim/                # Simulation source code
│   ├── config.py       # Catalogs, macro defaults, and scenario configs
│   ├── engine.py       # Single-user accounting and break-even math
│   ├── exhaustive.py   # Full grid search across factor combinations
│   ├── monte_carlo.py  # Cohort Monte Carlo engine with user segments
│   ├── sensitivity.py  # Stress tests and vulnerability corner analysis
│   ├── comparison.py   # Side-by-side reconciliation logic
│   ├── plots.py        # Executive chart generation
│   └── run.py          # Master runner script
├── tests/              # Verification test suite
│   ├── test_catalog_and_spec.py
│   ├── test_independent_accounting.py
│   ├── test_engines_and_invariants.py
│   └── test_outputs_and_plots_reconciliation.py
└── output/             # Generated simulation datasets and figures
    ├── figures/        # 7 publication-ready comparison charts
    └── *.csv           # Detailed output CSV tables
```

---

## Getting Started

### Prerequisites

Python 3.10 or higher. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Simulation

Execute the master simulation runner to calculate all grids, run Monte Carlo cohorts, and regenerate all output CSVs and charts:

```bash
python -m sim.run
```

All datasets write to `output/`, and generated figures save to `output/figures/`.

### Run the Test Suite

Run the full independent verification suite:

```bash
python -m unittest discover -s tests -v
```

All 21 tests check calculations against independent formulas, verify catalog definitions, and confirm chart text and margin formatting.

---

## Output Charts

The simulator outputs 7 publication-ready charts in `output/figures/`:

1. `when_the_fee_runs_out.png`: Chat volume break-even points across subscription plans.
2. `what_pictures_add.png`: Cost impact of media generation on plan margins.
3. `how_often_it_makes_money.png`: Percentage of profitable user scenarios across the exhaustive grid.
4. `pictures_cost_more_than_the_fee.png`: Media wholesale bill comparison against net subscription fees.
5. `priciest_model_vs_usual_mix.png`: Worst-case single-user loss compared against typical everyday margins.
6. `blocking_models.png`: Wholesale cost savings achieved by whitelisting specific models.
7. `monte_carlo_cohort_margins.png`: Margin distributions across 10,000 simulated subscriber cohorts.
