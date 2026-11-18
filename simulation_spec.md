# TASK SPECIFICATION: AI Platform Aggregator Per-User Tokenomics & Unit Economics Simulation Engine

> **Scope note, 6 October 2026.** The build is the Bangladesh-only simulator in `bd_info.md`. That file wins on conflict. This file remains the source for the margin formulas, which `bd_info.md` section 7 adapts to prepaid packs. Do not implement the generic subscription grid, the abstract Ultra/Standard/Economy price ranges, overage modes A/B/C, buyer segments, or a second cache discount. Language is Python with NumPy and Pandas. Grain is monthly totals per user. Monte Carlo, the sensitivity tables, and the P&L are still required, as specified in `bd_info.md`.

You are building a comprehensive, per-user managerial accounting and unit economics simulation for an AI aggregator platform (analogous to Cursor, Higgsfield, or OpenRouter). The platform provides unified access to multiple frontier and open-source models (LLMs, code generation, reasoning, image generation, multimodal video/audio) under subscription and metered tiers.

The formulas below were written for a generic aggregator. For this project they are applied to three Bangladeshi telco packs, one shared cohort of 100 users run on all three catalogs, and the price book and cost stack in `bd_info.md`.

---

## 1. THEORETICAL & BUSINESS ARCHITECTURE

An aggregator acts as a managed wholesale broker:
1. **Top-line Revenue:** Collected via flat recurring subscriptions ($/month), metered credit top-ups, and overage charges past allowances.
2. **Cost of Goods Sold (COGS):** Direct API token/generation expenditures incurred from upstream foundation model providers (OpenAI, Anthropic, Google, specialized media inference endpoints).
3. **The Margin Equation:** Gross Profit is determined by the spread between top-line subscription capture and wholesale provider bills, dictated by:
   - **Breakage & Inactive Subscriptions:** The gym membership effect where casual users under-consume their quota, subsidizing heavy power users.
   - **Prompt Caching & KV-Reuse:** Upstream discounts (up to 75% to 90%) applied when system prompts, file context trees, or conversation prefixes hit provider KV caches.
   - **Model Cascading & Smart Routing (FrugalGPT architecture):** Intercepting user queries and routing tasks dynamically between cheap utility models and expensive frontier models based on prompt complexity.
   - **Enterprise Tier Discounts:** Volume/negotiated discounts (0% to 30%) off provider list prices.
   - **Modality Blend:** Multi-modal tasks (image, video, deep reasoning tokens) consume non-linear compute units compared to standard short text completions.

---

## 2. SIMULATION SCOPE: 100-USER COHORT & DISTRIBUTIONS

Simulate an exact cohort of N = 100 users. Do NOT use static cohort-wide averages. Every individual user $i \in [1, 100]$ must sample their characteristics from realistic statistical distributions.

### A. User Segmentation & Volume Distribution (Power Law / Log-Normal)
User consumption follows an 80/20 Pareto distribution. Model raw consumption using a Log-Normal or Beta distribution:
- **Dormant / Low Usage (Breakage Cohort):** ~30% - 40% of users.
  - Active days/mo: 1 to 8 days.
  - Total Raw Text Tokens/mo: 100K to 1.5M tokens.
- **Moderate / Core Users:** ~40% - 50% of users.
  - Active days/mo: 15 to 22 days.
  - Total Raw Text Tokens/mo: 2M to 10M tokens.
- **Power Users / Heavy Autonomous Agents:** ~10% - 15% of users.
  - Active days/mo: 25 to 30 days.
  - Total Raw Text Tokens/mo: 15M to 60M+ tokens.

---

## 3. COMPREHENSIVE PARAMETER SPECTRA & VARIABLE RANGES

Implement the simulation such that the following parameter spaces can be configured or systematically swept (grid search or Monte Carlo iterations):

### Variable 1: Top-Line Packaging & Monetization Model
- `subscription_fee`: [$0, $10, $20, $40, $60, $200] per user / month.
- `included_frontier_allowance`: [0, 200, 500, 1000] fast/frontier requests OR token cap [$5, $10, $20 equivalent cost].
- `overage_policy`:
  - Mode A: Hard throttle / slow queue (COGS drops to $0 frontier cost after allowance).
  - Mode B: Metered overage billing (Platform charges raw cost + [10%, 20%, 30%] markup).
  - Mode C: Pure all-you-can-eat unconstrained (Aggregator absorbs all loss).
- `churn_refund_rate`: [0.0% to 3.0%] of monthly gross revenue.

### Variable 2: Raw Token Consumption & Modality Ratios (Per User)
For each user, break down demand into request types:
- **Input-to-Output Ratio:**
  - Range: `input_tokens : output_tokens` from 3:1 (short Q&A) to 15:1 (agentic code context / repo injection).
- **Text Tokens:** 100,000 to 60,000,000 tokens/user/month.
- **Specialized Reasoning Tokens (o1/o3/DeepSeek-R1 style):**
  - Range: 0% to 40% of total output tokens billed at hidden reasoning output rates.
- **Image Generation (Higgsfield / Midjourney style):**
  - Usage range: [0 to 150 images/user/month].
  - Upstream cost range: $0.015 to $0.080 per image (depending on resolution/steps).
- **Video Generation / Multimodal Compute:**
  - Usage range: [0 to 30 seconds of video generation/user/month].
  - Upstream cost range: $0.05 to $0.35 per second of generated video.

### Variable 3: Multi-Model Upstream Price Book (Per 1M Tokens)
Define upstream provider rate cards across at least 3 distinct performance tiers:
1. **Frontier Ultra Tier (e.g., GPT-5.6 Sol / Claude Opus / GPT-6 Astra):**
   - Uncached Input: $4.00 - $10.00 / 1M
   - Cached Input: $0.40 - $1.00 / 1M
   - Output: $20.00 - $50.00 / 1M
2. **Frontier Standard Tier (e.g., Claude Sonnet / GPT-4o / GPT-5.6 Terra):**
   - Uncached Input: $2.00 - $3.00 / 1M
   - Cached Input: $0.20 - $0.50 / 1M
   - Output: $10.00 - $15.00 / 1M
3. **Economy / Utility Tier (e.g., Claude Haiku / GPT-4o-mini / DeepSeek V3 / Distilled OSS):**
   - Uncached Input: $0.10 - $0.30 / 1M
   - Cached Input: $0.01 - $0.03 / 1M
   - Output: $0.40 - $1.20 / 1M

### Variable 4: Prompt Caching Dynamics
KV-cache reuse varies heavily by user interaction pattern:
- `cache_eligible_input_percentage`: Range [50% to 85%] of total input tokens (system prompts, git diffs, IDE indexing).
- `cache_hit_rate`: Range [0.0 to 0.90] (probability that eligible tokens match cached prefixes).
- `cache_write_premium`: Some providers charge a 1.25x premium on initial cache write vs regular input.
- `cache_read_discount`: Range [75% to 90%] discount on cached input reads.

### Variable 5: Smart Routing & Cascading Efficiency (FrugalGPT Spectrum)
Percentage of total user requests successfully cascaded down to lower-tier models without quality regression:
- `routing_efficiency_rate`:
  - 0% (Naive wrapper: 100% of queries go to chosen Frontier Ultra/Standard model).
  - Conservative: 30% Economy / 50% Standard / 20% Ultra.
  - Aggressive Cascading: 65% Economy / 25% Standard / 10% Ultra.
  - Extreme Auto-routing: 80% Economy / 15% Standard / 5% Ultra.

### Variable 6: Enterprise & Infrastructure Cost Factors
- `enterprise_volume_discount`: [0%, 10%, 15%, 20%, 25%] off total invoice from foundation labs.
- `router_and_vector_infra_overhead`: Fixed/variable compute cost to run internal embeddings, rerankers, and routing heuristics ($0.02 to $0.10 per 1M routed tokens).
- `payment_gateway_friction`: 2.9% + $0.30 per subscription transaction (plus 1.5% to 3.5% FX loss if operating across international currencies).

---

## 4. MATHEMATICAL FORMULATION REQUIRED IN ENGINE

For each user $i \in [1, 100]$:
1. $Rev_i = Subscription\_Fee_i + Overage\_Billed_i - Processing\_Fees_i$
2. Token split by model tier $m \in \{Economy, Standard, Ultra\}$ via routing matrix:
   $Tokens_{i, m} = Total\_Tokens_i \times Routing\_Weight_m$
3. Input tokens adjusted for caching:
   $Input\_Cost_{i, m} = \left[(Input_{i, m} \times (1 - HitRate_i) \times P_{uncached, m}) + (Input_{i, m} \times HitRate_i \times P_{cached, m})\right]$
4. $Output\_Cost_{i, m} = Output_{i, m} \times P_{output, m}$
5. $Modality\_Cost_i = (Images_i \times P_{image}) + (VideoSecs_i \times P_{videosec})$
6. $Raw\_COGS_i = \sum_{m} (Input\_Cost_{i, m} + Output\_Cost_{i, m}) + Modality\_Cost_i$
7. $Optimized\_COGS_i = (Raw\_COGS_i \times (1 - VolumeDiscount)) + Infra\_Overhead_i$
8. $Contribution\_Margin_i = Rev_i - Optimized\_COGS_i$
9. $Gross\_Margin\%_i = \frac{Contribution\_Margin_i}{Rev_i} \times 100$

Cohort Aggregates:
- $Cohort\_Revenue = \sum_{i=1}^{100} Rev_i$
- $Cohort\_COGS = \sum_{i=1}^{100} Optimized\_COGS_i$
- $Cohort\_Gross\_Margin = \frac{Cohort\_Revenue - Cohort\_COGS}{Cohort\_Revenue}$
- Identify % of Unprofitable Users (Users where $Contribution\_Margin_i < 0$) and Maximum Power User Loss.

---

## 5. CODE GENERATION DELIVERABLES

When building this simulation:
1. Create a modular, typed codebase (`SimulationConfig`, `UserGenerator`, `PricingEngine`, `MetricsVisualizer`).
2. Provide a CLI or interactive runner that can run:
   - A single deterministic baseline run (100 users, standard parameters).
   - A Monte Carlo simulation (1,000 iterations of 100-user cohorts) outputting mean, median, 5th percentile, and 95th percentile P&L outcomes.
   - Sensitivity Analysis tables: Matrix of `Cache Hit Rate` vs `Smart Routing Efficiency` showing resulting Blended Gross Margin %.
3. Include data visualization export (CSV dump or ASCII/rich terminal summary table) detailing segment-by-segment P&L (Dormant vs Core vs Power).