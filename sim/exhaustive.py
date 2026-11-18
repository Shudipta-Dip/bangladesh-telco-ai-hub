"""Full factor grid.

Text cost depends on model weights, input:output, cache hit, and cache write.
Pack usage depends on utilization and image use, and on video use for GP.
Those two groups are priced separately and then crossed by scaling.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sim.config import (
    FACTOR_STEP,
    FX_BUY,
    FX_WAR,
    IMAGE_USD,
    INFRA_USD_PER_1M,
    IO_RATIOS,
    MODELS,
    OPEN_MODELS,
    VIDEO_USD_PER_SEC,
    VOLUME_DISCOUNT,
    WEIGHT_PARTS,
    Pack,
    ScenarioConfig,
    banglalink_tokens_per_image,
    build_models,
    tokens_per_image_for_pack,
)
from sim.engine import effective_video_sec, net_revenue_bdt, recognized_and_cash
from sim.pricing import price_factors
from sim.simplex import mix_count, mix_matrix

N_BINS = 200_000
CHUNK = 2_000
DEFAULT_GATED_CHAT_VOLUMES = tuple(float(500_000 * i) for i in range(1, 12))  # 0.5M .. 5.5M (11 levels)


def factor_levels():
    n = int(round(1.0 / FACTOR_STEP))
    return np.linspace(0.0, 1.0, n + 1)


def _rates_for_weights(weights, miss, read, write, output, write_share: float = 0.0):
    """weights (chunk, models) -> USD per token, shape (chunk, n_io, n_cache)."""
    levels = factor_levels()
    io = np.array(IO_RATIOS, dtype=float)
    in_share = io / (io + 1.0)
    out_share = 1.0 / (io + 1.0)

    # levels is cache_hit fraction
    miss_share = np.maximum(0.0, (1.0 - levels)[:, None] - write_share)
    in_price = miss_share * miss[None, :] + levels[:, None] * read[None, :] + write_share * write[None, :]

    out_blend = weights @ output
    in_blend = weights @ in_price.T
    per_mtok = in_blend[:, None, :] * in_share[None, :, None] + out_blend[:, None, None] * out_share[None, :, None]
    return per_mtok / 1_000_000.0


def _rate_bound(miss, read, write, output):
    io = np.array(IO_RATIOS, dtype=float)
    in_share = io / (io + 1.0)
    out_share = 1.0 / (io + 1.0)
    max_in = np.maximum(miss, np.maximum(read, write))
    per_mtok = in_share[:, None] * max_in[None, :] + out_share[:, None] * output[None, :]
    return float(per_mtok.max() / 1_000_000.0)


def scan_rates(model_ids: List[str], scenario: Optional[ScenarioConfig] = None, write_share: float = 0.0):
    """Histogram every (mix, input:output, cache) rate for this catalog under a scenario."""
    sc = scenario or ScenarioConfig()
    models_dict = build_models(sc)
    parts = WEIGHT_PARTS
    mixes = mix_matrix(len(model_ids), parts)
    miss, read, write, output = price_factors(model_ids, models_dict)
    levels = factor_levels()
    miss_share = np.maximum(0.0, (1.0 - levels)[:, None] - write_share)
    in_price = miss_share * miss[None, :] + levels[:, None] * read[None, :] + write_share * write[None, :]

    bound = _rate_bound(miss, read, write, output)
    edges = np.linspace(0.0, bound * 1.001 + 1e-18, N_BINS + 1)
    hist = np.zeros(N_BINS, dtype=np.int64)
    total = 0
    sum_rate = 0.0
    sum_in_rate = 0.0
    min_rate = np.inf
    max_rate = 0.0
    worst = None
    best = None
    n_mixes = len(mixes)
    n_io = len(IO_RATIOS)
    for start in range(0, n_mixes, CHUNK):
        weights = mixes[start : start + CHUNK].astype(np.float64) / parts
        rates = _rates_for_weights(weights, miss, read, write, output, write_share=write_share)
        total += rates.size
        sum_rate += float(rates.sum())
        sum_in_rate += float((weights @ in_price.T).sum()) * n_io / 1_000_000.0
        chunk_min = float(rates.min())
        chunk_max = float(rates.max())
        eps = 1e-15
        if chunk_min <= min_rate + eps:
            # Among exact/IEEE-754 ties within eps, prefer the purest single-model mix.
            tied = np.argwhere(rates <= chunk_min + eps)
            cand_i = max(tied, key=lambda idx: int(mixes[start + idx[0]].max()))
            cand_mix = mixes[start + cand_i[0]]
            if chunk_min < min_rate - eps or (best is None or int(cand_mix.max()) > int(best[0].max())):
                min_rate = min(min_rate, chunk_min)
                best = (cand_mix.copy(), float(IO_RATIOS[cand_i[1]]), float(factor_levels()[cand_i[2]]), float(rates[tuple(cand_i)]))
        if chunk_max >= max_rate - eps:
            tied = np.argwhere(rates >= chunk_max - eps)
            cand_i = max(tied, key=lambda idx: int(mixes[start + idx[0]].max()))
            cand_mix = mixes[start + cand_i[0]]
            if chunk_max > max_rate + eps or (worst is None or int(cand_mix.max()) > int(worst[0].max())):
                max_rate = max(max_rate, chunk_max)
                worst = (cand_mix.copy(), float(IO_RATIOS[cand_i[1]]), float(factor_levels()[cand_i[2]]), float(rates[tuple(cand_i)]))
        hist += np.histogram(rates, bins=edges)[0]
    if int(hist.sum()) != total:
        raise RuntimeError("rate histogram dropped values")
    expected = mix_count(len(model_ids), parts) * len(IO_RATIOS) * len(factor_levels())
    if total != expected:
        raise RuntimeError(f"rate count {total} != {expected}")
    del mixes
    return {
        "model_ids": list(model_ids),
        "n_mixes": int(n_mixes),
        "hist": hist,
        "edges": edges,
        "count": total,
        "mean_rate": sum_rate / total,
        "mean_in_rate": sum_in_rate / total,
        "min_rate": float(min_rate),
        "max_rate": float(max_rate),
        "best": best,
        "worst": worst,
    }


def _cdf_fractions_at_or_below(hist, edges, thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    total = int(hist.sum())
    if total == 0:
        return np.full(thresholds.shape, np.nan, dtype=float)
    cdf = np.concatenate(([0], np.cumsum(hist)))
    covered = np.searchsorted(edges, thresholds, side="right")
    full = np.clip(covered - 1, 0, len(hist))
    out = cdf[full].astype(float) / total
    out = np.where(thresholds < edges[0], 0.0, out)
    out = np.where(thresholds >= edges[-1], 1.0, out)
    return out


def _cdf_fraction_at_or_below(hist, edges, threshold):
    if threshold < edges[0]:
        return 0.0
    if threshold >= edges[-1]:
        return 1.0
    covered = int(np.searchsorted(edges, threshold, side="right"))
    full = max(covered - 1, 0)
    total = int(hist.sum())
    if total == 0:
        return np.nan
    return float(hist[:full].sum()) / total


def _quantile_rate(hist, edges, q):
    cdf = np.cumsum(hist)
    total = int(cdf[-1])
    target = max(1, int(np.ceil(q * total)))
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = min(max(idx, 0), len(hist) - 1)
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def _text_cogs_bdt(text_tokens, rate, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    api = text_tokens * rate
    infra = text_tokens * (sc.infra_usd_per_1m / 1_000_000.0)
    return (api * (1.0 - sc.volume_discount) + infra) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)


def _usage_cells(pack: Pack, scenario: Optional[ScenarioConfig] = None, tokens_per_image: Optional[float] = None):
    sc = scenario or ScenarioConfig()
    levels = factor_levels()
    cells = []
    img_usd = pack.image_usd if pack.image_usd is not None else (sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd)
    tokens_per_img = tokens_per_image if tokens_per_image is not None else tokens_per_image_for_pack(pack, fx_buy=sc.fx_buy, image_usd=img_usd)

    for util in levels:
        image_levels = [0.0] if (pack.kind == "token_pool" and float(util) == 0.0) else levels
        for image_fraction in image_levels:
            budget = pack.tokens * float(util)
            if pack.kind == "token_pool":
                images = float(image_fraction) * (budget / tokens_per_img)
                text_tokens = budget * (1.0 - float(image_fraction))
                unused_images = 0.0
            else:
                images = float(image_fraction) * pack.images
                text_tokens = budget
                unused_images = (1.0 - float(image_fraction)) * pack.images
            image_bdt = images * img_usd * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
            recognized, cash = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
            net = float(net_revenue_bdt(recognized, btrc=sc.btrc, sof=sc.sof, mdr=sc.mdr, churn_refund=sc.churn_refund))
            cells.append(
                {
                    "utilization": float(util),
                    "image_fraction": float(image_fraction),
                    "text_tokens": float(text_tokens),
                    "images": float(images),
                    "unused_images": float(unused_images),
                    "image_bdt": float(image_bdt),
                    "cash_bdt": float(cash),
                    "recognized_bdt": float(recognized),
                    "net_bdt": net,
                    "breakage_memo_bdt": float(recognized) * (1.0 - float(util)),
                }
            )
    return cells


def _cell_stats(scan, cell, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    text_tokens = cell["text_tokens"]
    net = cell["net_bdt"]
    image_bdt = cell["image_bdt"]
    count = scan["count"]
    if text_tokens == 0:
        contribution = net - image_bdt
        margin = contribution / net if net > 0 else np.nan
        return {
            "contribution_mean": contribution,
            "contribution_min": contribution,
            "contribution_p05": contribution,
            "contribution_p50": contribution,
            "contribution_p95": contribution,
            "contribution_max": contribution,
            "gross_margin_p50": margin,
            "profitable_share": 1.0 if contribution >= 0 else 0.0,
            "distinct_cost_outcomes": 1,
            "text_cases": count,
        }
    scale_mean = _text_cogs_bdt(text_tokens, scan["mean_rate"], sc)

    def contrib(rate):
        return net - image_bdt - _text_cogs_bdt(text_tokens, rate, sc)

    p05_rate = _quantile_rate(scan["hist"], scan["edges"], 0.95)
    p50_rate = _quantile_rate(scan["hist"], scan["edges"], 0.50)
    p95_rate = _quantile_rate(scan["hist"], scan["edges"], 0.05)

    room = net - image_bdt
    per_token_infra = sc.infra_usd_per_1m / 1_000_000.0
    denom = text_tokens * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
    if denom == 0:
        rate_ceiling = np.inf
    else:
        rate_ceiling = (room / (sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)) - text_tokens * per_token_infra) / (text_tokens * (1.0 - sc.volume_discount))
    p50_contrib = contrib(p50_rate)
    return {
        "contribution_mean": net - image_bdt - scale_mean,
        "contribution_min": contrib(scan["max_rate"]),
        "contribution_p05": contrib(p05_rate),
        "contribution_p50": p50_contrib,
        "contribution_p95": contrib(p95_rate),
        "contribution_max": contrib(scan["min_rate"]),
        "gross_margin_p50": p50_contrib / net if net > 0 else np.nan,
        "profitable_share": _cdf_fraction_at_or_below(scan["hist"], scan["edges"], rate_ceiling),
        "distinct_cost_outcomes": count,
        "text_cases": count,
    }


def evaluate_token_pack(pack: Pack, scan, tokens_per_image: Optional[float] = None, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    rows = []
    for cell in _usage_cells(pack, scenario=sc, tokens_per_image=tokens_per_image):
        stats = _cell_stats(scan, cell, sc)
        row = {
            "platform": pack.platform,
            "pack": pack.name,
            "models": len(pack.models),
            "mixes": int(scan.get("n_mixes", len(scan.get("mixes", [])))),
            "text_cases": stats["text_cases"],
            "combinations": stats["text_cases"],
            **cell,
            **{k: stats[k] for k in stats if k != "text_cases"},
        }
        rows.append(row)
    return rows


def summarize_pack(rows, scan: Optional[Dict] = None, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    weight = np.array([row["combinations"] for row in rows], dtype=float)
    total = float(weight.sum())
    mean_contrib = float(np.average([row["contribution_mean"] for row in rows], weights=weight))
    profitable = float(np.average([row["profitable_share"] for row in rows], weights=weight))
    net_bdt = float(rows[0]["net_bdt"])
    mean_cogs = net_bdt - mean_contrib
    mean_breakage = float(np.average([row["breakage_memo_bdt"] for row in rows], weights=weight))

    if scan is not None and len(rows) > 0:
        q_rates = np.array([_quantile_rate(scan["hist"], scan["edges"], q) for q in np.linspace(0.0025, 0.9975, 200)])
        all_samples = []
        for row in rows:
            tt = row["text_tokens"]
            img_b = row["image_bdt"]
            net = row["net_bdt"]
            c_vals = net - img_b - _text_cogs_bdt(tt, q_rates, sc)
            all_samples.append(c_vals)
        all_c = np.concatenate(all_samples)
        p05_contrib = float(np.percentile(all_c, 5))
        p50_contrib = float(np.percentile(all_c, 50))
        p95_contrib = float(np.percentile(all_c, 95))
        min_contrib = float(min(row["contribution_min"] for row in rows))
        max_contrib = float(max(row["contribution_max"] for row in rows))
        margin_mean = float(mean_contrib / net_bdt) if net_bdt > 0 else np.nan
        margin_p50 = float(p50_contrib / net_bdt) if net_bdt > 0 else np.nan
    else:
        p05_contrib = float(np.percentile([row["contribution_p05"] for row in rows], 5))
        p50_contrib = float(np.median([row["contribution_p50"] for row in rows]))
        p95_contrib = float(np.percentile([row["contribution_p95"] for row in rows], 95))
        min_contrib = min(row["contribution_min"] for row in rows)
        max_contrib = max(row["contribution_max"] for row in rows)
        margin_mean = float(mean_contrib / net_bdt) if net_bdt > 0 else np.nan
        margin_p50 = float(p50_contrib / net_bdt) if net_bdt > 0 else np.nan

    return {
        "platform": rows[0]["platform"],
        "pack": rows[0]["pack"],
        "mixes": rows[0]["mixes"],
        "text_cases": rows[0]["text_cases"],
        "usage_cells": len(rows),
        "combinations": int(total),
        "mean_net_bdt": net_bdt,
        "mean_cogs_bdt": mean_cogs,
        "mean_contribution_bdt": mean_contrib,
        "p05_contribution_bdt": p05_contrib,
        "p50_contribution_bdt": p50_contrib,
        "p95_contribution_bdt": p95_contrib,
        "min_contribution_bdt": min_contrib,
        "max_contribution_bdt": max_contrib,
        "gross_margin_mean": margin_mean,
        "gross_margin_p50": margin_p50,
        "profitable_share": profitable,
        "mean_breakage_memo_bdt": mean_breakage,
    }


def gp_media_cells(pack: Pack, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    recognized, cash = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
    net = float(net_revenue_bdt(recognized, btrc=sc.btrc, sof=sc.sof, mdr=sc.mdr, churn_refund=sc.churn_refund))
    image_levels = factor_levels() if pack.images > 0 else [0.0]
    vid_cap = effective_video_sec(pack, sc)
    video_levels = factor_levels() if vid_cap > 0 else [0.0]
    audio_levels = factor_levels() if pack.audio > 0 else [0.0]

    img_price = pack.image_usd if pack.image_usd is not None else sc.image_usd
    vid_price = pack.video_usd_per_sec if pack.video_usd_per_sec is not None else sc.video_usd_per_sec
    aud_price = pack.audio_usd if pack.audio_usd is not None else 0.012
    files_cap = float(pack.files if (pack.platform == "banglalink" and pack.files > 0) else 0.0)

    cells = []
    # Full Cartesian 3D media product (images x video x audio): 1,331 cells when all three are present
    for image_fraction in image_levels:
        for video_fraction in video_levels:
            for audio_fraction in audio_levels:
                images = float(image_fraction) * pack.images
                video_sec = float(video_fraction) * vid_cap
                audio_items = float(audio_fraction) * pack.audio
                media_usd = images * img_price + video_sec * vid_price + audio_items * aud_price
                media_bdt = media_usd * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
                cells.append(
                    {
                        "platform": pack.platform,
                        "pack": pack.name,
                        "image_fraction": float(image_fraction),
                        "video_fraction": float(video_fraction),
                        "audio_fraction": float(audio_fraction),
                        "images": images,
                        "video_sec": video_sec,
                        "audio": audio_items,
                        "files": files_cap,
                        "media_bdt": float(media_bdt),
                        "cash_bdt": float(cash),
                        "recognized_bdt": float(recognized),
                        "net_bdt": net,
                        "room_bdt": net - float(media_bdt),
                    }
                )
    return cells


gated_media_cells = gp_media_cells


def _bdt_per_token(rate, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    return (rate * (1.0 - sc.volume_discount) + sc.infra_usd_per_1m / 1_000_000.0) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)


def _break_even(room_bdt, rate, scenario: Optional[ScenarioConfig] = None):
    unit = _bdt_per_token(rate, scenario)
    if unit <= 0:
        return np.nan
    return room_bdt / unit


def evaluate_gp_pack(pack: Pack, scan, scenario: Optional[ScenarioConfig] = None):
    sc = scenario or ScenarioConfig()
    rows = []
    p50_rate = _quantile_rate(scan["hist"], scan["edges"], 0.50)
    p05_rate = _quantile_rate(scan["hist"], scan["edges"], 0.05)
    p95_rate = _quantile_rate(scan["hist"], scan["edges"], 0.95)
    for cell in gp_media_cells(pack, sc):
        room = cell["room_bdt"]
        if room <= 0:
            # When room is negative, media alone wipes out net revenue. Zero text tokens can be served.
            be_min = be_p05 = be_p50 = be_p95 = be_max = 0.0
            media_already_loss = 1.0
        else:
            be_min = _break_even(room, scan["max_rate"], sc)
            be_p05 = _break_even(room, p95_rate, sc)
            be_p50 = _break_even(room, p50_rate, sc)
            be_p95 = _break_even(room, p05_rate, sc)
            be_max = _break_even(room, scan["min_rate"], sc)
            media_already_loss = 0.0
        rows.append(
            {
                "pack": pack.name,
                "mixes": int(scan.get("n_mixes", len(scan.get("mixes", [])))),
                "text_cases": scan["count"],
                "combinations": scan["count"],
                "usd_per_1m_p50": p50_rate * 1_000_000.0,
                "usd_per_1m_min": scan["min_rate"] * 1_000_000.0,
                "usd_per_1m_max": scan["max_rate"] * 1_000_000.0,
                "break_even_min": be_min,
                "break_even_p05": be_p05,
                "break_even_p50": be_p50,
                "break_even_p95": be_p95,
                "break_even_max": be_max,
                "media_already_unprofitable": media_already_loss,
                **cell,
            }
        )
    return rows


def gated_grid_profitability(
    gated_rows_or_df,
    scan,
    chat_tokens: Optional[float] = None,
    scenario: Optional[ScenarioConfig] = None,
    chat_volumes: Optional[tuple] = None,
):
    """Exact profitable share and scenario count across (media_cell x file_level x chat_volume x text_case)."""
    sc = scenario or ScenarioConfig()
    if hasattr(gated_rows_or_df, "iterrows"):
        rooms = gated_rows_or_df["room_bdt"].to_numpy(dtype=float)
        files_cap = float(gated_rows_or_df["files"].iloc[0]) if ("files" in gated_rows_or_df.columns and len(gated_rows_or_df) > 0) else 0.0
    else:
        rooms = np.array([float(r["room_bdt"]) for r in gated_rows_or_df], dtype=float)
        files_cap = float(gated_rows_or_df[0].get("files", 0.0)) if len(gated_rows_or_df) > 0 else 0.0
    if len(rooms) == 0:
        return 0.0, 0

    if chat_volumes is not None:
        vols = np.asarray(chat_volumes, dtype=float)
    elif chat_tokens is not None:
        vols = np.array([float(chat_tokens)], dtype=float)
    else:
        vols = np.asarray(DEFAULT_GATED_CHAT_VOLUMES, dtype=float)

    file_levels = factor_levels() if files_cap > 0 else np.array([0.0], dtype=float)
    mean_in_rate = float(scan.get("mean_in_rate", scan["mean_rate"] * 0.5))
    per_token_infra = sc.infra_usd_per_1m / 1_000_000.0
    mult = sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
    file_tokens = file_levels * files_cap * sc.bl_tokens_per_file
    file_bdts = (file_tokens * mean_in_rate * (1.0 - sc.volume_discount) + file_tokens * per_token_infra) * mult

    # Broadcast 3D grid: (n_media, n_file, n_vol)
    adj_rooms = rooms[:, None, None] - file_bdts[None, :, None]
    vols_3d = vols[None, None, :]
    denom = vols_3d * (1.0 - sc.volume_discount) * mult

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_caps = np.where(
            denom > 0,
            (adj_rooms / mult - vols_3d * per_token_infra) / (vols_3d * (1.0 - sc.volume_discount)),
            np.inf,
        )
    shares = _cdf_fractions_at_or_below(scan["hist"], scan["edges"], rate_caps)
    shares = np.where(adj_rooms <= 0, 0.0, shares)
    shares = np.where((adj_rooms > 0) & (vols_3d <= 0), 1.0, shares)

    total_combinations = int(rooms.size * file_levels.size * vols.size * scan["count"])
    return float(np.mean(shares)), total_combinations


def project_weights(open_ids, allowed_ids, weights, models_dict: Optional[Dict] = None):
    """Move weight on models the pack cannot serve onto its cheapest allowed model."""
    m_dict = models_dict or MODELS
    position = {model_id: i for i, model_id in enumerate(open_ids)}
    served = np.zeros((weights.shape[0], len(allowed_ids)), dtype=float)
    for j, m_id in enumerate(allowed_ids):
        if m_id in position:
            served[:, j] = weights[:, position[m_id]]
    cheapest = min(range(len(allowed_ids)), key=lambda i: (m_dict[allowed_ids[i]].output, m_dict[allowed_ids[i]].miss))
    missing = np.maximum(0.0, 1.0 - served.sum(axis=1))
    served[:, cheapest] += missing
    return served


def scan_gating(gp_pack_list: List[Pack], open_scan_models=None, scenario: Optional[ScenarioConfig] = None, write_share: float = 0.0):
    """One pass over the open-catalog lattice. Not added to the served GP grid."""
    sc = scenario or ScenarioConfig()
    models_dict = build_models(sc)
    open_ids = list(OPEN_MODELS if open_scan_models is None else open_scan_models)
    mixes = mix_matrix(len(open_ids), WEIGHT_PARTS)
    n_mixes = len(mixes)
    open_miss, open_read, open_write, open_out = price_factors(open_ids, models_dict)
    pack_state = []
    for pack in gp_pack_list:
        miss, read, write, output = price_factors(pack.models, models_dict)
        pack_state.append(
            {
                "pack": pack,
                "miss": miss,
                "read": read,
                "write": write,
                "output": output,
                "sum_avoided": 0.0,
                "cheaper": 0,
                "dearer": 0,
                "equal": 0,
                "count": 0,
                "min_avoided": np.inf,
                "max_avoided": -np.inf,
            }
        )
    gating_chunk = 2_000
    for start in range(0, n_mixes, gating_chunk):
        weights = mixes[start : start + gating_chunk].astype(np.float64) / WEIGHT_PARTS
        open_rates = _rates_for_weights(weights, open_miss, open_read, open_write, open_out, write_share=write_share)
        for state in pack_state:
            served_w = project_weights(open_ids, state["pack"].models, weights, models_dict)
            served_rates = _rates_for_weights(served_w, state["miss"], state["read"], state["write"], state["output"], write_share=write_share)
            avoided = (open_rates - served_rates) * 1_000_000.0
            state["count"] += avoided.size
            state["sum_avoided"] += float(avoided.sum())
            state["cheaper"] += int(np.sum(avoided > 1e-9))
            state["dearer"] += int(np.sum(avoided < -1e-9))
            state["equal"] += int(np.sum(np.abs(avoided) <= 1e-9))
            state["min_avoided"] = min(state["min_avoided"], float(avoided.min()))
            state["max_avoided"] = max(state["max_avoided"], float(avoided.max()))
            del served_w, served_rates, avoided
        del weights, open_rates
    del mixes
    rows = []
    for state in pack_state:
        count = state["count"]
        rows.append(
            {
                "pack": state["pack"].name,
                "open_mixes": int(n_mixes),
                "comparisons": count,
                "mean_avoided_usd_per_1m": state["sum_avoided"] / count,
                "min_avoided_usd_per_1m": state["min_avoided"],
                "max_avoided_usd_per_1m": state["max_avoided"],
                "share_whitelist_cheaper": state["cheaper"] / count,
                "share_whitelist_costlier": state["dearer"] / count,
                "share_same_cost": state["equal"] / count,
            }
        )
    return rows


def mix_label(model_ids, tenths, models_dict: Optional[Dict] = None):
    m_dict = models_dict or MODELS
    parts = [(m_dict[model_id].name, int(value)) for model_id, value in zip(model_ids, tenths) if int(value)]
    parts.sort(key=lambda item: -item[1])
    return " + ".join(f"{name} {value * 10}%" for name, value in parts)


def token_per_image(pack: Optional[Pack] = None, fx_buy: float = FX_BUY, image_usd: float = IMAGE_USD) -> float:
    from sim.engine import default_tokens_per_image
    return default_tokens_per_image(pack, fx_buy=fx_buy, image_usd=image_usd)
