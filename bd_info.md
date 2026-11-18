# BINDING SPEC: Bangladeshi Telco AI Hub Unit-Economics Simulator

This file is the build spec. `simulation_spec.md` stays as the formula source. Where the two differ, this file wins.

The simulator models retail AI subscription packs launched by Bangladesh's three major mobile network operators: **Banglalink (RYZE)**, **Robi (EveryAI)**, and **Grameenphone (OneAI)**. Each operator executes a distinct commercial packaging, model routing, VAT treatment, and media/file accounting architecture.

Stack: Python 3 with NumPy and Pandas. Monthly totals per user are the grain.
Research date for prices and Bangladesh charges: 6 October 2026.

---

## 1. STRATEGIC ARCHITECTURE BY TELCO OPERATOR

### A. Banglalink AI Hub (RYZE) — Gated Whitelists & Flat Subscriptions
* **Strategy:** Flat recurring monthly subscription packs with **Unlimited Chat** within rate limits, paired with strict model gating and hard quotas for media (images, videos, audio/music) and un-debited file uploads.
* **Transition Note:** Banglalink operates on **Gated Whitelists (`kind="gated"`, `tokens=None`)** across 5 commercial plans: Basic (৳100), Advanced (৳300), Pro (৳500), Creator (৳1,000), Legend (৳1,500).
* **VAT Baseline:** `bl_vat_mode = "inclusive"` (`Recognized = Sticker / 1.15`, reflecting direct digital/app checkout absorbing 15% VAT inside the sticker fee; talktime recharge and 50/50 split evaluated in `scenario_stress_test.csv`).
* **Media & File Treatment:**
  * Media generation (Images, Videos, Music/Audio) does **NOT** debit tokens from a balance; fixed quotas are absorbed by the flat fee.
  * **Video Clip Duration:** Plan cards state clip counts (`13` clips on Pro, `65` clips on Creator/Legend). Deterministic grid & worst-case corners use conservative **`5.0s/clip`** (`bl_video_sec_per_clip = 5.0`, yielding `65s` and `325s`); Monte Carlo draws continuous **`Uniform(4.0s, 8.0s)`** per user; stress suite also evaluates **`8.0s/clip`**.
  * **Independent Audio Axis:** `audio_fraction` varies independently in both the `3D` exhaustive media grid (`11 × 11 × 11 = 1,331` cells on Creator/Legend) and Monte Carlo zero-inflated Beta draws.
  * **Un-Debited File Context Tokens:** Banglalink's `5–80` file uploads do not debit a balance. Each uploaded file adds pure input context load (`bl_tokens_per_file = 20,000` input tokens/file deterministic; `Uniform(15,000, 25,000)` per user in Monte Carlo; `11` levels `0..100%` in exhaustive profitability), billed at the pack's weighted cache/miss input rate plus router infra overhead.
  * **Media Bill Risk:** Under app-inclusive net revenue, maxing out included media alone reaches **`1.41×` to `4.93×`** of net subscription revenue on Pro (`1.41×`), Legend (`3.49×`), and Creator (`4.93×`).

### B. Robi EveryAI — Mode-Based Tiered Intelligence & Token Pool Debit
* **Strategy:** Mode-based intelligent routing combined with document, slide, and code file generation. Robi is the **only operator** enforcing a prepaid token pool with direct token debit for both images and files.
* **VAT Baseline:** `robi_vat_mode = "inclusive"` (`Recognized = Sticker / 1.15`, with talktime recharge `"sticker_is_revenue"` and `50/50` split evaluated in stress tests).
* **4 Operating Modes:**
  1. **Saver:** Restricts access to `DeepSeek V4 Flash` and `Gemini 2.5 Flash Lite`.
  2. **Balanced:** Smart auto-routing (`70%` budget, `20%` mid, `10%` frontier).
  3. **Expert:** Unlocks all 8 text models (`20%` budget, `30%` mid, `50%` frontier).
  4. **Beast:** Uncapped output quotas, routing `50/50` to `Claude Haiku 4.5` and `Grok 4.3`.
* **Media & File Treatment:**
  * **Images and files DEDUCT directly from the token pool.** Standalone images (`Nano Banana 2 Lite` @ \$0.035, `GPT Image 2.5 Sunburst` @ \$0.050 -> midpoint \$0.0425) debit tokens at the pack's retail token rate:
    $$\text{TokensPerImage}_p = \frac{\text{ImageUSD} \times \text{fx\_buy}}{\text{PriceBDT}_p / \text{Tokens}_p}$$
  * Because debit matches wholesale cost at sticker price, full media conversion caps wholesale exposure identically across all 5 Robi packs at **`1.23×` net revenue** (`-Tk 18.7` per `Tk 100` paid under inclusive VAT + 6.5% BTRC/SOF wedge).
* **Every Robi Model (Exact 10 Models: 8 Text + 2 Image):**
  `Claude Haiku 4.5`, `DeepSeek V4 Flash`, `Gemini 2.5 Flash Lite`, `Gemini 3.5 Flash Lite`, `Kimi K2.6`, `GPT-5 Nano`, `Grok 4.1 Fast Non-Reasoning`, `Grok 4.3`, `Nano Banana 2 Lite`, `GPT Image 2.5 Sunburst`.

### C. Grameenphone OneAI — Gated Whitelists & Flat Recurring Subscriptions
* **Strategy:** Flat-fee prepaid packs with **NO text token allowance** (unlimited chat within rate limits).
* **VAT Baseline:** `gp_vat_mode = "added_on_top"` (`Cash = Sticker * 1.15`, `Recognized = Sticker`).
* **Catalog & Tiered Image Rates:**
  * Lite uses **`Grok 4.1 Fast` (`grok-4.1-fast-nr`, \$0.20/\$0.50)**; Plus unlocks **`Grok 4.3` (\$1.25/\$2.50)** alongside `GPT-5.4` and `Gemini 3.8 Flash`.
  * Explicit published-anchored deterministic image tiers: **\$0.035/image** on Free/Student/Lite (`Nano Banana 2 Lite` tier), **\$0.045/image** on Plus, and **\$0.050/image** on Pro (`Nano Banana 2` tier). Because GP does not publish image SKU names on consumer cards, Monte Carlo samples continuous **`gp_image_usd ~ Uniform($0.035, $0.050)`** per user.
* **Video Cap:** Hard **30-second** cap on GP Pro (`Veo 3.1 Fast 720p` @ \$0.100/sec).

---

## 2. DUAL-ANALYSIS & PERMANENT VERIFICATION ARCHITECTURE

1. **Exhaustive Factor Grid (Deterministic Perimeter):**
   * **Robi:** Full simplex lattice × 6 I:O ratios × 11 cache hits × 111 utilization/image cells (`142,476,048` scenarios/pack).
   * **Banglalink & GP:** Full whitelist simplex × 6 I:O ratios × 11 cache hits × full 3D media grid (`images × video × audio`, up to `1,331` cells) × `11` file levels (Banglalink) × **`11` chat volumes (`0.5M–5.5M` tokens)**.
2. **Empirical Monte Carlo Cohort Simulation:** `10,000` cohorts of `100` heterogeneous users (`1,000,000` users) sampled from continuous empirical priors (`Beta`, `Log-Mixture`, `Dirichlet` across all `27` active models, continuous GP image rates, `4–8s` BL clip lengths, independent audio & file fractions). Also evaluates **27-model active gating cost avoidance** across all `1,000,000` simulated users (`monte_carlo_summary.csv` & Chart 6 `blocking_models.png`), while retaining `scan_gating()` (`15`-model `OPEN_MODELS`) as the deterministic uniform reference (`gating_breakdown.csv`).
3. **Common Random Numbers (CRN) Stress Testing:** Reuses identical sampled cohorts across all `12` stress scenarios in `scenario_stress_test.csv` (`Baseline`, `Banglalink 8s/clip Video`, `BL/Robi Talktime Recharge`, `50/50 Channel VAT Split`, `Gemini 2027 Cliff`, `DeepSeek Peak`, `Claude 1h TTL`, `Long Context`, `10%/20% Volume Discounts`, `Router Infra $0.10/1M`, `Worst-Case Combined`).

---

## 3. PACK CATALOG & RATE STRUCTURE

### A. Banglalink RYZE AI Hub Pricing Plans

| Plan | Price (BDT) | Validity | AI Models | Chat Allowance | Files | Images | Videos | Audio | Work Mode |
|---|---:|---|---|---|---|---|---|---|---|
| Basic | ৳100 | 30 Days | 4 Models | Unlimited | Up to 5 | Up to 5 | N/A | N/A | N/A |
| Advanced | ৳300 | 30 Days | 5 Models | Unlimited | Up to 15 | Up to 15 | N/A | N/A | N/A |
| Pro | ৳500 | 30 Days | 10 Models | Unlimited | Up to 25 | Up to 54 | Up to 13 (65s @ 5s) | N/A | N/A |
| Creator | ৳1,000 | 30 Days | 24 Models | Unlimited | Up to 30 | Up to 272 | Up to 65 (325s @ 5s) | Up to 389 | Included |
| Legend | ৳1,500 | 30 Days | 24 Models | Unlimited | Up to 80 | Up to 272 | Up to 65 (325s @ 5s) | Up to 389 | Included |

#### Banglalink Model Availability Matrix

| Model | Category / Type | Basic | Advanced | Pro | Creator | Legend |
|---|---|:---:|:---:|:---:|:---:|:---:|
| GPT 6 | LLM / Chat | ✅ | ✅ | ✅ | ✅ | ❌ |
| GPT 6.1 Terra | LLM / Chat | ❌ | ❌ | ❌ | ❌ | ✅ |
| Gemini 3.1 | LLM / Chat | ✅ | ✅ | ❌ | ✅ | ❌ |
| Gemini 3.8 | LLM / Chat | ❌ | ❌ | ✅ | ❌ | ❌ |
| Gemini 3.8 Flash | LLM / Chat | ❌ | ❌ | ❌ | ✅ | ✅ |
| Claude Haiku 5.5 | LLM / Chat | ❌ | ❌ | ✅ | ✅ | ❌ |
| Claude Sonnet 5 | LLM / Chat | ❌ | ❌ | ❌ | ✅ | ✅ |
| Grok 4.3 | LLM / Chat | ❌ | ✅ | ✅ | ✅ | ❌ |
| Grok 4.6 | LLM / Chat | ❌ | ❌ | ❌ | ✅ | ✅ |
| Grok Imagine | Image Model | ❌ | ❌ | ❌ | ❌ | ✅ |
| DeepSeek V4.1 | LLM / Chat | ✅ | ✅ | ✅ | ✅ | ❌ |
| DeepSeek V4 Pro | LLM / Chat | ❌ | ❌ | ❌ | ❌ | ✅ |
| Qwen 3.8 Max | LLM / Chat | ❌ | ❌ | ❌ | ✅ | ✅ |
| Perplexity Sonar Pro | LLM / Chat | ❌ | ❌ | ❌ | ✅ | ✅ |
| Mistral Large 2512 | LLM / Chat | ❌ | ❌ | ❌ | ✅ | ✅ |
| FLUX Schnell | Image Model | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX 2 Pro | Image Model | ❌ | ❌ | ✅ | ✅ | ✅ |
| GPT Image 2 | Image Model | ❌ | ❌ | ❌ | ✅ | ✅ |
| Nano Banana 2 Lite | Image Model | ❌ | ❌ | ✅ | ❌ | ❌ |
| Nano Banana 2 | Image Model | ❌ | ❌ | ❌ | ✅ | ✅ |
| Seedream 5 Pro | Image Model | ❌ | ❌ | ❌ | ✅ | ✅ |
| Seedance 1 Pro Fast | Video Model | ❌ | ❌ | ✅ | ✅ | ✅ |
| Hailuo 02 Fast | Video Model | ❌ | ❌ | ❌ | ✅ | ✅ |
| Wan 2.2 5B Fast | Video Model | ❌ | ❌ | ✅ | ✅ | ✅ |
| Lightricks LTX 2.5 | Video Model | ❌ | ❌ | ❌ | ❌ | ✅ |
| P-Video Draft | Video Model | ❌ | ❌ | ❌ | ✅ | ✅ |
| MiniMax Music 1.5 | Audio / Music | ❌ | ❌ | ❌ | ✅ | ✅ |
| Flux Music | Audio / Music | ❌ | ❌ | ❌ | ✅ | ✅ |
| Meta MusicGen | Audio / Music | ❌ | ❌ | ❌ | ✅ | ✅ |
| ElevenLabs Voice | Audio / Voice | ❌ | ❌ | ❌ | ✅ | ✅ |
| ElevenLabs Music | Audio / Music | ❌ | ❌ | ❌ | ❌ | ✅ |

### B. Robi EveryAI (Monthly Token Pool with Image/File Debit)
| Plan | Price (BDT) | Token Pool | Implied Retail Rate (BDT/1M) | Validity | Allowed Modes |
|---|---:|---:|---:|---|---|
| Pack 1 | 99 | 1,000,000 | Tk 99.0 | 30 days | Saver, Balanced |
| Pack 2 | 149 | 2,000,000 | Tk 74.5 | 30 days | Saver, Balanced |
| Pack 3 | 349 | 5,000,000 | Tk 69.8 | 30 days | Saver, Balanced, Expert |
| Pack 4 | 599 | 10,000,000 | Tk 59.9 | 30 days | Saver, Balanced, Expert, Beast |
| Pack 5 | 999 | 20,000,000 | Tk 50.0 | 30 days | Saver, Balanced, Expert, Beast |

### C. Grameenphone OneAI (Gated Whitelists)
| Plan | Price (BDT) | Allowed Text Models | Image Quota & Benchmark Rate | Video Quota |
|---|---:|---|---|---|
| Free | 0 | GPT-5 Nano, Gemini 3 Flash | 0 (\$0.035) | None |
| Student | 71 | GPT-5 Nano, Gemini 3 Flash | 10 (\$0.035) | None |
| Lite | 99 | DeepSeek V4 Flash, Gemini 3 Flash, **Grok 4.1 Fast**, GPT-5 Nano | 30 (\$0.035) | None |
| Plus | 299 | DeepSeek V4 Flash, Gemini 3 Flash, **Grok 4.3**, GPT-5 Nano, Gemini 3.8 Flash, GPT-5 Mini, GPT-5.4, Kimi K2.6 | 50 (\$0.045) | None |
| Pro | 799 | Claude Sonnet 4, Claude Sonnet 5, Claude Opus 5, GPT-5.6 Sol, Grok 4.6, DeepSeek V4 Pro | 100 (\$0.050) | 30 seconds hard cap |

---

## 4. MASTER UPSTREAM PRICE BOOK (USD per 1M tokens unless noted)

Research date: 6 October 2026. Published short-context list rates.

| Model ID | Public Name | Platform Presence | Role / Tier | Miss | Cache Read | Cache Write | Output |
|---|---|---|---|---:|---:|---:|---:|
| `gpt-5-nano` | GPT-5 Nano | Robi, GP | Economy | 0.05 | 0.005 | — | 0.40 |
| `gpt-6` | GPT 6 | Banglalink | Economy | 0.15 | 0.015 | — | 0.60 |
| `gpt-5-mini` | GPT-5 Mini | GP | Mid | 0.25 | 0.025 | — | 2.00 |
| `gpt-5.4` | GPT-5.4 | GP | Flagship | 2.50 | 0.250 | — | 15.00 |
| `gpt-5.6-sol` | GPT-5.6 Sol | GP | Flagship | 4.00 | 0.400 | 5.00 | 20.00 |
| `gpt-6.1-terra` | GPT 6.1 Terra | Banglalink | Flagship | 2.00 | 0.200 | — | 12.00 |
| `claude-haiku-4.5` | Claude Haiku 4.5 | Robi | Frontier / Economy | 1.00 | 0.100 | 1.25 | 5.00 |
| `claude-haiku-5.5` | Claude Haiku 5.5 | Banglalink | Economy / Mid | 1.00 | 0.100 | 1.25 | 5.00 |
| `claude-sonnet-4` | Claude Sonnet 4 | GP | Flagship | 3.00 | 0.300 | 3.75 | 15.00 |
| `claude-sonnet-5` | Claude Sonnet 5 | Banglalink, GP | Mid | 2.00 | 0.200 | 2.50 | 10.00 |
| `claude-opus-5` | Claude Opus 5 | GP | Flagship | 5.00 | 0.500 | 6.25 | 25.00 |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | Robi | Economy | 0.10 | 0.010 | — | 0.40 |
| `gemini-3.1` | Gemini 3.1 | Banglalink | Economy | 0.10 | 0.010 | — | 0.40 |
| `gemini-3.5-flash-lite` | Gemini 3.5 Flash Lite | Robi | Mid | 0.30 | 0.030 | — | 2.50 |
| `gemini-3-flash` | Gemini 3 Flash | GP | Mid | 0.50 | 0.050 | — | 3.00 |
| `gemini-3.8` | Gemini 3.8 | Banglalink | Mid | 0.75 | 0.075 | — | 3.75 |
| `gemini-3.8-flash` | Gemini 3.8 Flash | Banglalink, GP | Mid | 0.75 | 0.075 | — | 3.75 |
| `grok-4.1-fast-nr` | Grok 4.1 Fast NR | Robi, **GP** (Lite) | Economy | 0.20 | 0.050 | — | 0.50 |
| `grok-4.3` | Grok 4.3 | Banglalink, Robi, **GP** (Plus) | Mid / Frontier | 1.25 | 0.200 | — | 2.50 |
| `grok-4.6` | Grok 4.6 | Banglalink, GP | Mid / Frontier | 2.00 | 0.500 | — | 6.00 |
| `deepseek-v4-flash` | DeepSeek V4 Flash | Robi, GP | Economy | 0.15 | 0.003 | — | 0.60 |
| `deepseek-v4.1` | DeepSeek V4.1 | Banglalink | Economy | 0.15 | 0.003 | — | 0.60 |
| `deepseek-v4-pro` | DeepSeek V4 Pro | Banglalink, GP | Mid | 0.66 | 0.022 | — | 1.98 |
| `kimi-k2.6` | Kimi K2.6 | Robi, GP | Mid | 0.95 | 0.160 | — | 4.00 |
| `qwen3.8-max` | Qwen 3.8 Max | Banglalink | Mid | 2.00 | 2.000 | — | 6.00 |
| `perplexity-sonar-pro` | Perplexity Sonar Pro | Banglalink | Flagship | 3.00 | 0.500 | — | 15.00 |
| `mistral-large-2512` | Mistral Large 2512 | Banglalink | Mid | 2.00 | 0.500 | — | 6.00 |

#### Multimodal Media Upstream Pricing
* **Image Generation:**
  * `FLUX Schnell`: \$0.003 / image
  * `FLUX 2 Pro`: \$0.040 / image
  * `GPT Image 2`: \$0.040 / image
  * `Nano Banana 2 Lite`: \$0.035 / image (Robi & GP Student/Lite anchor)
  * `Nano Banana 2`: \$0.050 / image (GP Pro anchor)
  * `Seedream 5 Pro`: \$0.035 / image
  * `Grok Imagine`: \$0.040 / image
* **Video Generation:**
  * `Seedance 1 Pro Fast`: \$0.060 / sec
  * `Wan 2.2 5B Fast`: \$0.040 / sec
  * `Hailuo 02 Fast`: \$0.080 / sec
  * `Lightricks LTX 2.5`: \$0.080 / sec
  * `P-Video Draft`: \$0.050 / sec
  * `Veo 3.1 Fast 720p` (GP): \$0.100 / sec
* **Audio & Voice Generation:**
  * `MiniMax Music 1.5`, `Flux Music`, `Meta MusicGen`, `ElevenLabs Voice`: \$0.012 / generation
  * `ElevenLabs Music`: \$0.015 / generation

---

## 5. BANGLADESH OPERATIONAL COST STACK

* **FX Rates:** Interbank ask $\text{fx\_buy} = 123.30$ BDT/USD; Weighted average $\text{fx\_war} = 123.20$ BDT/USD.
* **BTRC Gross Revenue Share:** 5.5% of recognized revenue across all operators.
* **Social Obligation Fund (SOF):** 1.0% of recognized revenue across all operators.
* **Collection Channel MDR:** Default 0.0% (Bangla QR MDR removed Oct 2026). Scenario sweeps 1.0% and 1.85%.
* **Per-Operator VAT Mode:**
  * GP OneAI (`gp_vat_mode`): `added_on_top` (`Cash = Sticker * 1.15`; `Recognized = Sticker`).
  * Banglalink (`bl_vat_mode`): `inclusive` baseline (`Recognized = Sticker / 1.15`); stress tests cover `sticker_is_revenue` & `mixed_50_50`.
  * Robi (`robi_vat_mode`): `inclusive` baseline (`Recognized = Sticker / 1.15`); stress tests cover `sticker_is_revenue` & `mixed_50_50`.
* **Router & Vector Infra:** \$0.02 per 1M routed tokens (MC uniform `0.02..0.06`, stress \$0.08–\$0.10).
* **Bank Remittance TT Commission:** Default 0.0% (MC uniform `0.0%..1.5%`, sweep up to `2.5%`).

---

## 6. UNIT MARGIN AND COGS FORMULAS

### A. Net Recognized Revenue
$$\text{NetRevBDT} = \text{RecognizedBDT} \times (1 - 0.055 - 0.01) \times (1 - \text{mdr}) \times (1 - \text{churn})$$

### B. Token Pool Operator Accounting (Robi)
1. **Token Pool Debit for Media:**
   $$\text{TokensPerImage}_p = \frac{\text{ImageUSD} \times \text{fx\_buy}}{\text{PriceBDT}_p / \text{Tokens}_p}$$
2. **Usage Allocation:**
   $$\text{ImageTokens} = f_{\text{img}} \times \text{Budget}, \quad \text{TextTokens} = \text{Budget} - \text{ImageTokens}$$
3. **Total COGS & Margin:**
   $$\text{TextCOGS\_BDT} = \left(\text{TextTokens} \times \text{Rate}_{\text{token}} \times (1 - d) + \frac{\text{InfraUSD}}{10^6} \times \text{TextTokens}\right) \times \text{fx\_buy} \times (1 + r_{\text{tt}})$$
   $$\text{MediaCOGS\_BDT} = \text{Images} \times \text{ImageUSD} \times (1 - d) \times \text{fx\_buy} \times (1 + r_{\text{tt}})$$
   $$\text{ContributionBDT} = \text{NetRevBDT} - (\text{TextCOGS\_BDT} + \text{MediaCOGS\_BDT})$$

### C. Gated Flat Subscription Accounting (Banglalink & GP)
1. **Media Wholesale Bill (with variable BL clip duration):**
   $$\text{MediaUSD} = (\text{Images} \times \text{UnitImgUSD}) + (\text{VideoSec} \times \text{UnitVidUSD}) + (\text{Audio} \times \text{UnitAudUSD})$$
   $$\text{MediaBDT} = \text{MediaUSD} \times (1 - d) \times \text{fx\_buy} \times (1 + r_{\text{tt}})$$
2. **Un-Debited File Context Load (Banglalink only):**
   $$\text{FileTokens} = f_{\text{file}} \times \text{Files}_p \times \text{TokensPerFile}$$
   $$\text{FileBDT} = \left(\text{FileTokens} \times \text{InRate}_{\text{token}} \times (1 - d) + \frac{\text{InfraUSD}}{10^6} \times \text{FileTokens}\right) \times \text{fx\_buy} \times (1 + r_{\text{tt}})$$
3. **Room Remaining & Break-Even:**
   $$\text{RoomBDT} = \text{NetRevBDT} - \text{MediaBDT} - \text{FileBDT}$$
   $$\text{BreakEvenTokens} = \begin{cases} 0.0 & \text{if } \text{RoomBDT} \le 0 \\ \frac{\text{RoomBDT}}{\text{BDTPerToken}} & \text{if } \text{RoomBDT} > 0 \end{cases}$$
4. **Total COGS & Contribution:**
   $$\text{TotalCOGS\_BDT} = \text{ChatCOGS\_BDT} + \text{MediaBDT} + \text{FileBDT}, \quad \text{ContributionBDT} = \text{NetRevBDT} - \text{TotalCOGS\_BDT}$$
