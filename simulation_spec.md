# Unit Economics Specification

This document details the accounting formulas and user modeling logic used to simulate retail AI subscriptions.

## 1. Revenue Accounting

A subscriber pays a monthly fee in Bangladeshi Taka (BDT). Net recognized revenue depends on channel tax and regulatory deductions:

1. **Value Added Tax (VAT):** Digital checkout channels absorb 15% VAT inside the sticker price. Plans that add VAT at checkout preserve the full sticker price.
2. **Regulatory Fees:** The Bangladesh Telecommunication Regulatory Commission (BTRC) collects a 5.5% revenue share and a 1.0% Social Obligation Fund (SOF) fee, totaling 6.5%.
3. **Net Revenue Equation:**
   $$\text{Net Revenue} = \text{Recognized Revenue} \times (1 - 0.055 - 0.010) \times (1 - \text{MDR}) \times (1 - \text{Refunds})$$

## 2. Cost of Goods Sold (COGS)

Wholesale API bills from model providers determine product costs:

1. **Text Tokens:**
   Calculated from prompt input tokens, cache read tokens, cache write tokens, and completion output tokens.
   $$\text{Input Cost} = (1 - h - w) \times P_{\text{miss}} + h \times P_{\text{read}} + w \times P_{\text{write}}$$
   Where $h$ is the cache hit rate and $w$ is the cache write share.
2. **Media Generation:**
   Wholesale costs for images, video clips, and audio tracks billed per generation or per second.
3. **Infrastructure Overhead:**
   A routing and embedding charge ($0.02 to $0.06 per 1M tokens) added to total token volume.
4. **Foreign Exchange:**
   US Dollar costs convert to BDT using the interbank exchange rate (123.30 BDT/USD) plus bank transfer commission (0.0% to 1.5%).

## 3. User Segmentation

Subscriber behavior follows three main tiers:

- **Dormant Users (35%):** Low activity (1 to 8 days per month), consuming under 1.5M text tokens and few media requests.
- **Core Users (50%):** Regular activity (15 to 22 days per month), consuming 1M to 3M text tokens and moderate media requests.
- **Power Users (15%):** Heavy activity (25 to 30 days per month), consuming 5M to 15M text tokens with high input-to-output context ratios.

## 4. Profitability Metrics

For each user $i$:
$$\text{Contribution Margin}_i = \text{Net Revenue}_i - \text{Total COGS}_i$$
$$\text{Gross Margin \%}_i = \frac{\text{Contribution Margin}_i}{\text{Net Revenue}_i} \times 100$$

A cohort of 100 users aggregates individual subscriber margins to calculate average portfolio profit and identify unprofitable customer rates.
