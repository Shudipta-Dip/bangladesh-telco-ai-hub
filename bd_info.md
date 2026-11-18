# Bangladesh Telco AI Plans: Tariff and Cost Reference

This document compiles retail subscription plans, foundation model rate cards, and local regulatory costs for consumer AI services in Bangladesh.

Three operators offer consumer AI plans:
- **Banglalink (RYZE):** Monthly subscriptions with model whitelists, media quotas, and un-debited file uploads.
- **Robi (EveryAI):** Prepaid monthly token pools with routing modes and token deductions for images and files.
- **Grameenphone (OneAI):** Monthly subscriptions with model whitelists and image allowances, charging VAT on top of checkout.

---

## 1. Operator Commercial Packaging

### A. Banglalink AI Hub (RYZE)
* **Structure:** Recurring monthly subscriptions with unlimited chat on whitelisted models, fixed quotas for media (images, video clips, audio tracks), and un-debited file uploads.
* **Plans:** Basic (৳100), Advanced (৳300), Pro (৳500), Creator (৳1,000), Legend (৳1,500).
* **VAT:** In-app digital checkouts absorb 15% VAT inside the sticker fee (`Recognized = Sticker / 1.15`).
* **Video Clip Duration:** 5.0 seconds per clip baseline (65s on Pro, 325s on Creator and Legend). Monte Carlo samples 4.0 to 8.0 seconds per clip.
* **File Uploads:** 5 to 80 uploaded files. Each file adds 20,000 input context tokens billed at weighted cache input rates plus server infrastructure costs.
* **Wholesale Exposure:** Because media quotas do not deduct from a token balance, heavy media usage on Creator (৳1,000) reaches 4.93x net subscription revenue.

### B. Robi EveryAI
* **Structure:** Prepaid monthly token budgets where text chat, images, and files draw from one token pool.
* **Plans:** Pack 1 (৳99, 1M tokens), Pack 2 (৳149, 2M tokens), Pack 3 (৳349, 5M tokens), Pack 4 (৳599, 10M tokens), Pack 5 (৳999, 20M tokens).
* **VAT:** In-app checkouts absorb 15% VAT inside the sticker fee (`Recognized = Sticker / 1.15`).
* **Routing Modes:**
  1. Saver: Routes to DeepSeek V4 Flash and Gemini 2.5 Flash Lite. Available on all packs.
  2. Balanced: Routes across budget (70%), mid (20%), and frontier (10%) models. Available on all packs.
  3. Expert: Unlocks all 8 text models (20% budget, 30% mid, 50% frontier). Available on Pack 3, 4, and 5.
  4. Beast: Routes 50/50 between Claude Haiku 4.5 and Grok 4.3. Available on Pack 4 and 5.
* **Media Deductions:** Images and files deduct tokens from the pool using the pack implied retail rate:
  $$\text{TokensPerImage}_p = \frac{\text{ImageUSD} \times \text{fx\_buy}}{\text{PriceBDT}_p / \text{Tokens}_p}$$
* **Loss Ceiling:** Because deductions match wholesale cost at sticker price, wholesale media cost cannot exceed collected revenue. Maximum loss stops at -18.7% of sticker price (-৳18.70 per ৳100) under inclusive VAT and 6.5% regulatory fees.

### C. Grameenphone OneAI
* **Structure:** Monthly subscriptions with unlimited text on whitelisted models and fixed image quotas.
* **VAT:** Grameenphone adds 15% VAT on top of checkout prices (`Cash = Sticker * 1.15`, `Recognized = Sticker`).
* **Plans:** Free (৳0), Student (৳71), Lite (৳99), Plus (৳299), Pro (৳799).
* **Key Allocations:**
  * Free: GPT-5 Nano and Gemini 3 Flash. 0 images.
  * Student (৳71): GPT-5 Nano and Gemini 3 Flash. 10 images ($0.035 each).
  * Lite (৳99): Adds Grok 4.1 Fast Non-Reasoning and DeepSeek V4 Flash. 30 images ($0.035 each).
  * Plus (৳299): Adds Grok 4.3, GPT-5.4, Gemini 3.8 Flash, Kimi K2.6. 50 images ($0.045 each).
  * Pro (৳799): Claude Sonnet 4 and 5, Claude Opus 5, GPT-5.6 Sol, Grok 4.6, DeepSeek V4 Pro. 100 images ($0.050 each) and a 30-second video cap ($0.10/sec).

---

## 2. Plan Catalogs and Quotas

### Banglalink RYZE Plans
| Plan | Price (BDT) | Validity | Allowed Models | Chat | Files | Images | Video Clips (5s) | Audio Tracks |
|---|---:|---|---|---|---|---|---|---|
| Basic | ৳100 | 30 Days | 3 Models | Unlimited | 5 | 5 | 0 | 0 |
| Advanced | ৳300 | 30 Days | 4 Models | Unlimited | 15 | 15 | 0 | 0 |
| Pro | ৳500 | 30 Days | 5 Models | Unlimited | 25 | 54 | 13 (65s) | 0 |
| Creator | ৳1,000 | 30 Days | 11 Models | Unlimited | 30 | 272 | 65 (325s) | 389 |
| Legend | ৳1,500 | 30 Days | 8 Models | Unlimited | 80 | 272 | 65 (325s) | 389 |

#### Banglalink Model Distribution
| Model | Category | Basic | Advanced | Pro | Creator | Legend |
|---|---|:---:|:---:|:---:|:---:|:---:|
| GPT 6 | Chat | Yes | Yes | Yes | Yes | No |
| GPT 6.1 Terra | Chat | No | No | No | No | Yes |
| Gemini 3.1 | Chat | Yes | Yes | No | Yes | No |
| Gemini 3.8 | Chat | No | No | Yes | No | No |
| Gemini 3.8 Flash | Chat | No | No | No | Yes | Yes |
| Claude Haiku 5.5 | Chat | No | No | Yes | Yes | No |
| Claude Sonnet 5 | Chat | No | No | No | Yes | Yes |
| Grok 4.3 | Chat | No | Yes | Yes | Yes | No |
| Grok 4.6 | Chat | No | No | No | Yes | Yes |
| Grok Imagine | Image | No | No | No | No | Yes |
| DeepSeek V4.1 | Chat | Yes | Yes | Yes | Yes | No |
| DeepSeek V4 Pro | Chat | No | No | No | No | Yes |
| Qwen 3.8 Max | Chat | No | No | No | Yes | Yes |
| Perplexity Sonar Pro | Chat | No | No | No | Yes | Yes |
| Mistral Large 2512 | Chat | No | No | No | Yes | Yes |
| FLUX Schnell | Image | Yes | Yes | Yes | Yes | Yes |
| FLUX 2 Pro | Image | No | No | Yes | Yes | Yes |
| GPT Image 2 | Image | No | No | No | Yes | Yes |
| Nano Banana 2 Lite | Image | No | No | Yes | No | No |
| Nano Banana 2 | Image | No | No | No | Yes | Yes |
| Seedream 5 Pro | Image | No | No | No | Yes | Yes |
| Seedance 1 Pro Fast | Video | No | No | Yes | Yes | Yes |
| Hailuo 02 Fast | Video | No | No | No | Yes | Yes |
| Wan 2.2 5B Fast | Video | No | No | Yes | Yes | Yes |
| Lightricks LTX 2.5 | Video | No | No | No | No | Yes |
| P-Video Draft | Video | No | No | No | Yes | Yes |
| MiniMax Music 1.5 | Audio | No | No | No | Yes | Yes |
| Flux Music | Audio | No | No | No | Yes | Yes |
| Meta MusicGen | Audio | No | No | No | Yes | Yes |
| ElevenLabs Voice | Voice | No | No | No | Yes | Yes |
| ElevenLabs Music | Audio | No | No | No | No | Yes |

### Robi EveryAI Plans
| Plan | Price (BDT) | Token Pool | Implied Rate (BDT/1M) | Validity | Allowed Modes |
|---|---:|---:|---:|---|---|
| Pack 1 | ৳99 | 1,000,000 | ৳99.00 | 30 days | Saver, Balanced |
| Pack 2 | ৳149 | 2,000,000 | ৳74.50 | 30 days | Saver, Balanced |
| Pack 3 | ৳349 | 5,000,000 | ৳69.80 | 30 days | Saver, Balanced, Expert |
| Pack 4 | ৳599 | 10,000,000 | ৳59.90 | 30 days | Saver, Balanced, Expert, Beast |
| Pack 5 | ৳999 | 20,000,000 | ৳49.95 | 30 days | Saver, Balanced, Expert, Beast |

### Grameenphone OneAI Plans
| Plan | Price (BDT) | Allowed Chat Models | Image Quota and Benchmark | Video Quota |
|---|---:|---|---|---|
| Free | ৳0 | GPT-5 Nano, Gemini 3 Flash | 0 images ($0.035) | None |
| Student | ৳71 | GPT-5 Nano, Gemini 3 Flash | 10 images ($0.035) | None |
| Lite | ৳99 | DeepSeek V4 Flash, Gemini 3 Flash, Grok 4.1 Fast, GPT-5 Nano | 30 images ($0.035) | None |
| Plus | ৳299 | DeepSeek V4 Flash, Gemini 3 Flash, Grok 4.3, GPT-5 Nano, Gemini 3.8 Flash, GPT-5 Mini, GPT-5.4, Kimi K2.6 | 50 images ($0.045) | None |
| Pro | ৳799 | Claude Sonnet 4, Claude Sonnet 5, Claude Opus 5, GPT-5.6 Sol, Grok 4.6, DeepSeek V4 Pro | 100 images ($0.050) | 30 seconds ($0.10/sec) |

---

## 3. Upstream Model Price Book

List rates in USD per 1M tokens (October 2026):

| Model ID | Public Name | Platform | Tier | Miss | Cache Read | Cache Write | Output |
|---|---|---|---|---:|---:|---:|---:|
| `gpt-5-nano` | GPT-5 Nano | Robi, GP | Economy | 0.05 | 0.005 | None | 0.40 |
| `gpt-6` | GPT 6 | Banglalink | Economy | 0.15 | 0.015 | None | 0.60 |
| `gpt-5-mini` | GPT-5 Mini | GP | Mid | 0.25 | 0.025 | None | 2.00 |
| `gpt-5.4` | GPT-5.4 | GP | Flagship | 2.50 | 0.250 | None | 15.00 |
| `gpt-5.6-sol` | GPT-5.6 Sol | GP | Flagship | 4.00 | 0.400 | 5.00 | 20.00 |
| `gpt-6.1-terra` | GPT 6.1 Terra | Banglalink | Flagship | 2.00 | 0.200 | None | 12.00 |
| `claude-haiku-4.5` | Claude Haiku 4.5 | Robi | Frontier / Economy | 1.00 | 0.100 | 1.25 | 5.00 |
| `claude-haiku-5.5` | Claude Haiku 5.5 | Banglalink | Economy / Mid | 1.00 | 0.100 | 1.25 | 5.00 |
| `claude-sonnet-4` | Claude Sonnet 4 | GP | Flagship | 3.00 | 0.300 | 3.75 | 15.00 |
| `claude-sonnet-5` | Claude Sonnet 5 | Banglalink, GP | Mid | 2.00 | 0.200 | 2.50 | 10.00 |
| `claude-opus-5` | Claude Opus 5 | GP | Flagship | 5.00 | 0.500 | 6.25 | 25.00 |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | Robi | Economy | 0.10 | 0.010 | None | 0.40 |
| `gemini-3.1` | Gemini 3.1 | Banglalink | Economy | 0.10 | 0.010 | None | 0.40 |
| `gemini-3.5-flash-lite` | Gemini 3.5 Flash Lite | Robi | Mid | 0.30 | 0.030 | None | 2.50 |
| `gemini-3-flash` | Gemini 3 Flash | GP | Mid | 0.50 | 0.050 | None | 3.00 |
| `gemini-3.8` | Gemini 3.8 | Banglalink | Mid | 0.75 | 0.075 | None | 3.75 |
| `gemini-3.8-flash` | Gemini 3.8 Flash | Banglalink, GP | Mid | 0.75 | 0.075 | None | 3.75 |
| `grok-4.1-fast-nr` | Grok 4.1 Fast NR | Robi, GP | Economy | 0.20 | 0.050 | None | 0.50 |
| `grok-4.3` | Grok 4.3 | Banglalink, Robi, GP | Mid / Frontier | 1.25 | 0.200 | None | 2.50 |
| `grok-4.6` | Grok 4.6 | Banglalink, GP | Mid / Frontier | 2.00 | 0.500 | None | 6.00 |
| `deepseek-v4-flash` | DeepSeek V4 Flash | Robi, GP | Economy | 0.15 | 0.003 | None | 0.60 |
| `deepseek-v4.1` | DeepSeek V4.1 | Banglalink | Economy | 0.15 | 0.003 | None | 0.60 |
| `deepseek-v4-pro` | DeepSeek V4 Pro | Banglalink, GP | Mid | 0.66 | 0.022 | None | 1.98 |
| `kimi-k2.6` | Kimi K2.6 | Robi, GP | Mid | 0.95 | 0.160 | None | 4.00 |
| `qwen3.8-max` | Qwen 3.8 Max | Banglalink | Mid | 2.00 | 2.000 | None | 6.00 |
| `perplexity-sonar-pro` | Perplexity Sonar Pro | Banglalink | Flagship | 3.00 | 0.500 | None | 15.00 |
| `mistral-large-2512` | Mistral Large 2512 | Banglalink | Mid | 2.00 | 0.500 | None | 6.00 |

### Media Generation Costs
- **Images:** FLUX Schnell ($0.003), FLUX 2 Pro ($0.040), GPT Image 2 ($0.040), Nano Banana 2 Lite ($0.035), Nano Banana 2 ($0.050), Seedream 5 Pro ($0.035), Grok Imagine ($0.040).
- **Video:** Seedance 1 Pro Fast ($0.060/s), Wan 2.2 5B Fast ($0.040/s), Hailuo 02 Fast ($0.080/s), Lightricks LTX 2.5 ($0.080/s), P-Video Draft ($0.050/s), Veo 3.1 Fast ($0.100/s).
- **Audio and Voice:** MiniMax Music 1.5 ($0.012), Flux Music ($0.012), Meta MusicGen ($0.012), ElevenLabs Voice ($0.012), ElevenLabs Music ($0.015).

---

## 4. Local Cost Stack and Regulations

- **Exchange Rates:** 123.30 BDT per USD base rate.
- **BTRC Revenue Share:** 5.5% of recognized revenue.
- **Social Obligation Fund (SOF):** 1.0% of recognized revenue.
- **Combined Government Take:** 6.5% of recognized revenue.
- **VAT Modes:**
  - Added on top (Grameenphone): Customer pays 15% VAT on top of sticker. Recognized revenue equals sticker price.
  - Inclusive (Banglalink and Robi): Digital checkout absorbs 15% VAT inside sticker price. Recognized revenue equals sticker divided by 1.15.
- **Server and Routing Overhead:** $0.02 per 1M routed tokens base cost ($0.02 to $0.06 in simulations).
- **Bank Transfer Commission:** 0.0% base cost (0.0% to 1.5% in simulations).

---

## 5. Margin and Accounting Formulas

### Net Recognized Revenue
$$\text{Net Revenue} = \text{Recognized Revenue} \times (1 - 0.055 - 0.01) \times (1 - \text{MDR}) \times (1 - \text{Churn})$$

### Robi Token Pool Accounting
1. Token Pool Debit for Images:
   $$\text{TokensPerImage}_p = \frac{\text{ImageUSD} \times \text{fx\_buy}}{\text{PriceBDT}_p / \text{Tokens}_p}$$
2. Usage Allocation:
   $$\text{ImageTokens} = f_{\text{img}} \times \text{Budget}, \quad \text{TextTokens} = \text{Budget} - \text{ImageTokens}$$
3. Contribution:
   $$\text{Contribution} = \text{Net Revenue} - (\text{Text Cost} + \text{Image Cost})$$

### Flat Subscription Accounting (Banglalink and GP)
1. Wholesale Media Bill:
   $$\text{Media Bill USD} = (\text{Images} \times P_{\text{img}}) + (\text{Video Sec} \times P_{\text{vid}}) + (\text{Audio} \times P_{\text{aud}})$$
2. Un-debited File Context Load (Banglalink):
   $$\text{File Tokens} = f_{\text{file}} \times \text{Files}_p \times \text{TokensPerFile}$$
3. Total Contribution:
   $$\text{Contribution} = \text{Net Revenue} - (\text{Chat Cost} + \text{Media Cost} + \text{File Cost})$$
