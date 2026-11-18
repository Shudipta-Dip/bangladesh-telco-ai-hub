"""Model mixes and per-token upstream cost.

Supports cache-write pricing, scenario overrides, bank TT commission, and pack-specific image debit.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from sim.config import (
    AUTO_TIER,
    BANK_TT_COMMISSION_PCT,
    FX_BUY,
    IMAGE_USD,
    INFRA_USD_PER_1M,
    MANUAL_TIER,
    MODELS,
    VOLUME_DISCOUNT,
    Model,
    banglalink_tokens_per_image as bl_tokens_calc,
)

TIER_ORDER = ("economy", "mid", "flagship")


def _by_tier(model_ids, models_dict: Optional[Dict[str, Model]] = None):
    m_dict = models_dict or MODELS
    grouped = {tier: [] for tier in TIER_ORDER}
    for model_id in model_ids:
        model = m_dict[model_id]
        grouped[model.tier].append(model)
    return grouped


def _cheapest_tier(grouped):
    present = [tier for tier in TIER_ORDER if grouped[tier]]
    return min(present, key=lambda tier: min(model.output for model in grouped[tier]))


def tier_mix(model_ids, shares, models_dict: Optional[Dict[str, Model]] = None):
    """Split tier weights across the models in each tier.

    A tier with no model on this catalog folds onto the cheapest tier that exists.
    """
    m_dict = models_dict or MODELS
    grouped = _by_tier(model_ids, m_dict)
    present = [tier for tier in TIER_ORDER if grouped[tier]]
    if not present:
        raise ValueError("catalog is empty")
    cheapest = _cheapest_tier(grouped)
    folded = {tier: 0.0 for tier in TIER_ORDER}
    for tier, weight in shares.items():
        if grouped[tier]:
            folded[tier] += weight
        else:
            folded[cheapest] += weight
    raw = {}
    for tier in present:
        if folded[tier] <= 0:
            continue
        each = folded[tier] / len(grouped[tier])
        for model in grouped[tier]:
            raw[model.id] = raw.get(model.id, 0.0) + each
    total = sum(raw.values())
    return {model_id: weight / total for model_id, weight in raw.items()}


def one_hot(model_ids, pick, models_dict: Optional[Dict[str, Model]] = None):
    m_dict = models_dict or MODELS
    if pick == "cheapest":
        chosen = min((m_dict[i] for i in model_ids), key=lambda model: (model.output, model.miss))
    elif pick == "dearest":
        # Worst-case corner is I:O = 3:1 (75% uncached input, 25% output) with 0% cache reuse.
        chosen = max(
            (m_dict[i] for i in model_ids),
            key=lambda model: (0.75 * model.miss + 0.25 * model.output, model.output, model.miss),
        )
    else:
        raise ValueError(pick)
    return {chosen.id: 1.0}


def routed_components(model_ids, models_dict: Optional[Dict[str, Model]] = None):
    return (
        tier_mix(model_ids, MANUAL_TIER, models_dict),
        tier_mix(model_ids, AUTO_TIER, models_dict),
    )


def blend_weights(manual_weights, auto_weights, manual_share):
    """manual_share is a scalar or an array. Returns arrays aligned to a fixed key order."""
    keys = list(dict.fromkeys([*manual_weights.keys(), *auto_weights.keys()]))
    manual = np.array([manual_weights.get(key, 0.0) for key in keys], dtype=float)
    auto = np.array([auto_weights.get(key, 0.0) for key in keys], dtype=float)
    share = np.asarray(manual_share, dtype=float)
    weights = share[..., None] * manual + (1.0 - share[..., None]) * auto
    return keys, weights


def price_factors(model_ids, models_dict: Optional[Dict[str, Model]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (miss, read, write, output) arrays for the given model ids."""
    m_dict = models_dict or MODELS
    miss = np.array([m_dict[i].miss for i in model_ids], dtype=float)
    read = np.array([m_dict[i].read_price for i in model_ids], dtype=float)
    write = np.array([m_dict[i].write_price for i in model_ids], dtype=float)
    output = np.array([m_dict[i].output for i in model_ids], dtype=float)
    return miss, read, write, output


def usd_per_token(weights, io_ratio, cache_hit, miss, read, output, write=None, cache_write_share=0.0):
    """Upstream text USD per pack token, accounting for cache miss, read, and write pricing.

    weights: (..., n_models)
    io_ratio, cache_hit: broadcastable to leading shape
    cache_write_share: fraction of input tokens that are initial cache writes
    """
    io_ratio = np.asarray(io_ratio, dtype=float)
    cache_hit = np.asarray(cache_hit, dtype=float)
    write_share = np.asarray(cache_write_share, dtype=float)

    if write is None:
        write = miss

    in_share = io_ratio / (io_ratio + 1.0)
    out_share = 1.0 / (io_ratio + 1.0)

    miss_share = np.maximum(0.0, 1.0 - cache_hit - write_share)
    in_price = miss_share[..., None] * miss + cache_hit[..., None] * read + write_share[..., None] * write

    blended = (weights * (in_share[..., None] * in_price + out_share[..., None] * output)).sum(axis=-1)
    return blended / 1_000_000.0


def usd_per_input_token(weights, cache_hit, miss, read, write=None, cache_write_share=0.0):
    """Upstream text USD per pure INPUT token (e.g., uploaded file context injection)."""
    cache_hit = np.asarray(cache_hit, dtype=float)
    write_share = np.asarray(cache_write_share, dtype=float)
    if write is None:
        write = miss
    miss_share = np.maximum(0.0, 1.0 - cache_hit - write_share)
    in_price = miss_share[..., None] * miss + cache_hit[..., None] * read + write_share[..., None] * write
    return (weights * in_price).sum(axis=-1) / 1_000_000.0


def text_cogs_bdt(
    tokens,
    usd_per_tok,
    volume_discount: float = VOLUME_DISCOUNT,
    infra_usd_per_1m: float = INFRA_USD_PER_1M,
    fx_buy: float = FX_BUY,
    bank_tt_commission_pct: float = BANK_TT_COMMISSION_PCT,
):
    api_usd = tokens * usd_per_tok
    infra_usd = tokens * (infra_usd_per_1m / 1_000_000.0)
    cogs_bdt = (api_usd * (1.0 - volume_discount) + infra_usd) * fx_buy * (1.0 + bank_tt_commission_pct)
    return api_usd, infra_usd, cogs_bdt


def banglalink_tokens_per_image(entry_price_bdt, entry_tokens, fx_buy: float = FX_BUY, image_usd: float = IMAGE_USD):
    """Legacy helper for entry pack rate."""
    retail_bdt_per_token = entry_price_bdt / entry_tokens
    return (image_usd * fx_buy) / retail_bdt_per_token
