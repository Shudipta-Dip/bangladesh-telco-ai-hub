"""Sensitivity analysis, utilization-band P&L, scenario stress tests, and vulnerability corners.

Generates:
1. sensitivity_cache_manual.csv (Cache Hit Rate vs Manual Share)
2. sensitivity_io_dearest.csv (I:O Ratio vs Weight on Dearest Model)
3. grid_by_band.csv (Cohort P&L by utilization bands: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0])
4. vulnerability_corners.csv (Exact worst-case boundary points for all packs)
5. scenario_stress_test.csv (Impact of Gemini 2027 cliff, DeepSeek peak, discounts, etc.)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sim.config import (
    FACTOR_STEP,
    FX_BUY,
    FX_WAR,
    IO_RATIOS,
    MODELS,
    OPEN_MODELS,
    Pack,
    ScenarioConfig,
    banglalink_tokens_per_image,
    build_models,
    gp_packs,
    token_packs,
    tokens_per_image_for_pack,
)
from sim.engine import (
    MixBook,
    contribution_at_volume,
    effective_video_sec,
    gp_result,
    net_revenue_bdt,
    recognized_and_cash,
    token_pool_result,
)
from sim.pricing import one_hot, price_factors, text_cogs_bdt, usd_per_input_token, usd_per_token

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def generate_sensitivity_cache_manual(packs: List[Pack], scenario: Optional[ScenarioConfig] = None) -> pd.DataFrame:
    """Matrix of Cache Hit Rate vs Manual Share showing Blended Gross Margin %."""
    sc = scenario or ScenarioConfig()
    cache_levels = np.linspace(0.0, 1.0, 11)
    manual_levels = np.linspace(0.0, 1.0, 11)
    rows = []

    for pack in packs:
        if pack.price_bdt <= 0:
            continue
        book = MixBook(pack.models, sc)
        unit_img_price = sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd
        tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=sc.fx_buy, image_usd=unit_img_price) if pack.kind == "token_pool" else None

        for c_hit in cache_levels:
            for m_share in manual_levels:
                if pack.kind == "token_pool":
                    # Assume median utilization (0.5) and median chat I:O (5:1)
                    res = token_pool_result(
                        pack=pack,
                        scenario_name="routed",
                        utilization=0.5,
                        cache_hit=c_hit,
                        manual_share=m_share,
                        io_ratio=5.0,
                        image_fraction=0.3,
                        tokens_per_image=tokens_per_img,
                        scenario=sc,
                    )
                    margin = float(res["gross_margin"])
                    contrib = float(res["contribution_bdt"])
                else:
                    # Gated pack (Banglalink & GP)
                    res = gp_result(
                        pack=pack,
                        scenario_name="routed",
                        cache_hit=c_hit,
                        manual_share=m_share,
                        io_ratio=5.0,
                        image_fraction=0.3,
                        video_fraction=0.2,
                        audio_fraction=0.3,
                        file_fraction=0.3,
                        scenario=sc,
                    )
                    # Evaluate at 1M chat tokens including media, un-debited files, infra, volume discount, and TT commission
                    net = float(res["net_bdt"])
                    contrib = float(
                        contribution_at_volume(
                            float(res["gated_usd_per_1m"]) / 1_000_000.0,
                            1_000_000.0,
                            net,
                            float(res["media_bdt"]) + float(res["file_bdt"]),
                            sc,
                        )
                    )
                    margin = contrib / net if net > 0 else np.nan

                rows.append(
                    {
                        "platform": pack.platform,
                        "pack": pack.name,
                        "cache_hit_rate": round(c_hit, 2),
                        "manual_share": round(m_share, 2),
                        "gross_margin": margin,
                        "contribution_bdt": contrib,
                    }
                )

    df = pd.DataFrame(rows)
    return df


def generate_sensitivity_io_dearest(packs: List[Pack], scenario: Optional[ScenarioConfig] = None) -> pd.DataFrame:
    """Matrix of I:O Ratio vs Dearest Model Weight showing Blended Gross Margin %."""
    sc = scenario or ScenarioConfig()
    io_levels = list(IO_RATIOS)
    dearest_levels = np.linspace(0.0, 1.0, 11)
    rows = []

    for pack in packs:
        if pack.price_bdt <= 0:
            continue
        book = MixBook(pack.models, sc)
        unit_img_price = pack.image_usd if pack.image_usd is not None else (sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd)
        tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=sc.fx_buy, image_usd=unit_img_price) if pack.kind == "token_pool" else None

        for io in io_levels:
            for dear_w in dearest_levels:
                keys, weights = book.dearest_blend(dear_w)
                miss, read, write, output = price_factors(keys, book.models_dict)
                per_tok = usd_per_token(weights, io, 0.5, miss, read, output, write=write)
                in_per_tok = usd_per_input_token(weights, 0.5, miss, read, write=write)

                if pack.kind == "token_pool":
                    budget = pack.tokens * 0.5
                    images = 0.2 * (budget / tokens_per_img)
                    text_tokens = budget - images * tokens_per_img

                    _, _, text_bdt = text_cogs_bdt(text_tokens, per_tok, sc.volume_discount, sc.infra_usd_per_1m, sc.fx_buy, sc.bank_tt_commission_pct)
                    image_bdt = images * unit_img_price * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
                    cogs = text_bdt + image_bdt
                    recognized, cash = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
                    net = float(net_revenue_bdt(recognized, sc.btrc, sc.sof, sc.mdr, sc.churn_refund))
                    contrib = net - cogs
                    margin = contrib / net if net > 0 else np.nan
                else:
                    # Gated pack (Banglalink & GP)
                    img_price = pack.image_usd if pack.image_usd is not None else sc.image_usd
                    vid_price = pack.video_usd_per_sec if pack.video_usd_per_sec is not None else sc.video_usd_per_sec
                    aud_price = pack.audio_usd if pack.audio_usd is not None else 0.012
                    vid_cap = effective_video_sec(pack, sc)
                    images = 0.2 * pack.images
                    video_sec = 0.1 * vid_cap
                    audio_items = 0.2 * pack.audio
                    media_bdt = (images * img_price + video_sec * vid_price + audio_items * aud_price) * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
                    file_tokens = 0.2 * pack.files * sc.bl_tokens_per_file if (pack.platform == "banglalink" and pack.files > 0) else 0.0
                    _, _, file_bdt = text_cogs_bdt(file_tokens, float(in_per_tok), sc.volume_discount, sc.infra_usd_per_1m, sc.fx_buy, sc.bank_tt_commission_pct)
                    recognized, _ = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
                    net = float(net_revenue_bdt(recognized, sc.btrc, sc.sof, sc.mdr, sc.churn_refund))
                    # Chat tokens: 1M tokens
                    _, _, text_bdt = text_cogs_bdt(1_000_000.0, per_tok, sc.volume_discount, sc.infra_usd_per_1m, sc.fx_buy, sc.bank_tt_commission_pct)
                    cogs = text_bdt + media_bdt + file_bdt
                    contrib = net - cogs
                    margin = contrib / net if net > 0 else np.nan

                rows.append(
                    {
                        "platform": pack.platform,
                        "pack": pack.name,
                        "io_ratio": float(io),
                        "dearest_model_weight": round(dear_w, 2),
                        "gross_margin": margin,
                        "contribution_bdt": contrib,
                    }
                )

    df = pd.DataFrame(rows)
    return df


def generate_vulnerability_corners(packs: List[Pack], scenario: Optional[ScenarioConfig] = None) -> pd.DataFrame:
    """Explicitly calculates the vulnerability corner as specified in bd_info.md §2:

    - utilization: 1.0
    - manual share: 1.0, 100% on the dearest allowed model
    - cache hit: 0.0, cache write: 0.0
    - input:output: 3:1 (output-heavy, costly)
    - image fraction: 1.0
    - video fraction: 1.0 (5s/clip conservative baseline on Banglalink, 30s on GP Pro)
    - audio fraction: 1.0 (when present)
    - file fraction: 1.0 (Banglalink context input load when files > 0)
    """
    sc = scenario or ScenarioConfig()
    models_dict = build_models(sc)
    rows = []

    for pack in packs:
        book = MixBook(pack.models, sc)
        dearest_id = book.dearest_id
        dearest_model = models_dict[dearest_id]

        miss, read, write, output = price_factors([dearest_id], models_dict)
        # 100% on dearest, I:O = 3:1, cache_hit = 0.0
        per_token_rate = usd_per_token([1.0], 3.0, 0.0, miss, read, output, write=write, cache_write_share=0.0)

        recognized, cash = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
        net_bdt = float(net_revenue_bdt(recognized, sc.btrc, sc.sof, sc.mdr, sc.churn_refund))

        if pack.kind == "token_pool":
            unit_img_price = pack.image_usd if pack.image_usd is not None else (sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd)
            tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=sc.fx_buy, image_usd=unit_img_price)
            # Compare 100% dearest text model vs 100% image conversion and take the true worst case.
            _, _, cogs_all_text = text_cogs_bdt(pack.tokens, per_token_rate, sc.volume_discount, sc.infra_usd_per_1m, sc.fx_buy, sc.bank_tt_commission_pct)
            imgs_all = pack.tokens / tokens_per_img
            cogs_all_img = imgs_all * unit_img_price * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
            if float(cogs_all_text) >= float(cogs_all_img):
                images = 0.0
                text_tokens = pack.tokens
                total_cogs = float(cogs_all_text)
            else:
                images = float(imgs_all)
                text_tokens = 0.0
                total_cogs = float(cogs_all_img)
            video_sec = 0.0
            file_tokens = 0.0
        else:
            # Gated pack (Banglalink & GP): full media quota + un-debited files (BL) + 2M chat tokens at dearest model
            img_price = pack.image_usd if pack.image_usd is not None else sc.image_usd
            vid_price = pack.video_usd_per_sec if pack.video_usd_per_sec is not None else sc.video_usd_per_sec
            aud_price = pack.audio_usd if pack.audio_usd is not None else 0.012
            vid_cap = effective_video_sec(pack, sc)

            images = pack.images
            video_sec = vid_cap
            audio_items = pack.audio
            media_bdt = (images * img_price + video_sec * vid_price + audio_items * aud_price) * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
            text_tokens = 2_000_000.0  # reference benchmark volume
            _, _, text_bdt = text_cogs_bdt(text_tokens, per_token_rate, sc.volume_discount, sc.infra_usd_per_1m, sc.fx_buy, sc.bank_tt_commission_pct)
            file_tokens = pack.files * sc.bl_tokens_per_file if (pack.platform == "banglalink" and pack.files > 0) else 0.0
            file_in_rate = float(miss[0]) / 1_000_000.0
            _, _, file_bdt = text_cogs_bdt(file_tokens, file_in_rate, sc.volume_discount, sc.infra_usd_per_1m, sc.fx_buy, sc.bank_tt_commission_pct)
            total_cogs = float(text_bdt) + float(media_bdt) + float(file_bdt)

        contrib = net_bdt - total_cogs
        margin = contrib / net_bdt if net_bdt > 0 else np.nan

        rows.append(
            {
                "platform": pack.platform,
                "pack": pack.name,
                "price_bdt": pack.price_bdt,
                "net_bdt": net_bdt,
                "dearest_model": dearest_model.name,
                "dearest_rate_usd_per_1m": float(per_token_rate * 1_000_000.0),
                "text_tokens_sent": text_tokens,
                "file_tokens_sent": float(file_tokens),
                "images_generated": images,
                "video_sec_generated": video_sec,
                "total_cogs_bdt": total_cogs,
                "contribution_bdt": contrib,
                "gross_margin": margin,
            }
        )

    return pd.DataFrame(rows)


def generate_grid_by_utilization_bands(packs: List[Pack], scenario: Optional[ScenarioConfig] = None) -> pd.DataFrame:
    """P&L grouped into the 4 specified bands: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]."""
    sc = scenario or ScenarioConfig()
    from sim.exhaustive import evaluate_token_pack, scan_rates

    bands = [
        ("[0, 0.25)", 0.0, 0.25),
        ("[0.25, 0.50)", 0.25, 0.50),
        ("[0.50, 0.75)", 0.50, 0.75),
        ("[0.75, 1.0]", 0.75, 1.01),
    ]

    scans = {}
    rows = []
    for pack in packs:
        if pack.kind != "token_pool":
            continue
        key = tuple(pack.models)
        if key not in scans:
            scans[key] = scan_rates(pack.models, sc)
        scan = scans[key]
        unit_img_price = sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd
        tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=sc.fx_buy, image_usd=unit_img_price)
        eval_rows = evaluate_token_pack(pack, scan, tokens_per_img, sc)
        df_eval = pd.DataFrame(eval_rows)

        for band_name, low, high in bands:
            sub = df_eval[(df_eval["utilization"] >= low) & (df_eval["utilization"] < high)]
            if len(sub) == 0:
                continue
            weight = sub["combinations"].to_numpy(dtype=float)
            total_weight = weight.sum()
            mean_contrib = float(np.average(sub["contribution_mean"], weights=weight))
            min_contrib = float(sub["contribution_min"].min())
            max_contrib = float(sub["contribution_max"].max())
            profitable = float(np.average(sub["profitable_share"], weights=weight))
            net_bdt = float(sub["net_bdt"].iloc[0])
            breakage = float(np.average(sub["breakage_memo_bdt"], weights=weight))
            cogs_bdt = net_bdt - mean_contrib

            rows.append(
                {
                    "platform": pack.platform,
                    "pack": pack.name,
                    "band": band_name,
                    "top_line_collected_bdt": float(sub["cash_bdt"].iloc[0]),
                    "recognized_bdt": float(sub["recognized_bdt"].iloc[0]),
                    "net_usd": net_bdt / sc.fx_war,
                    "net_bdt": net_bdt,
                    "cogs_bdt": cogs_bdt,
                    "cogs_usd": cogs_bdt / sc.fx_buy,
                    "contribution_bdt": mean_contrib,
                    "min_contribution_bdt": min_contrib,
                    "max_contribution_bdt": max_contrib,
                    "gross_margin": mean_contrib / net_bdt if net_bdt > 0 else np.nan,
                    "profitable_share": profitable,
                    "breakage_memo_bdt": breakage,
                }
            )

    return pd.DataFrame(rows)


def run_scenario_stress_test(packs: List[Pack]) -> pd.DataFrame:
    """Stress tests packs against regulatory, product, and upstream pricing shifts under CRN."""
    scenarios = [
        ScenarioConfig(name="Baseline"),
        ScenarioConfig(name="Banglalink 8s/clip Video", bl_video_sec_per_clip=8.0),
        ScenarioConfig(
            name="BL/Robi Talktime Recharge (No App VAT Deduction)",
            bl_vat_mode="sticker_is_revenue",
            robi_vat_mode="sticker_is_revenue",
        ),
        ScenarioConfig(
            name="BL/Robi 50/50 Channel VAT Split",
            bl_vat_mode="mixed_50_50",
            robi_vat_mode="mixed_50_50",
        ),
        ScenarioConfig(name="Gemini 2027 Cliff (+100% price)", gemini_38_price="list_2027"),
        ScenarioConfig(name="DeepSeek Peak Window (2x price)", deepseek_window="peak"),
        ScenarioConfig(name="Claude 1h Cache Write TTL", claude_cache_ttl="1h"),
        ScenarioConfig(name="Long Context (>200k tokens)", long_context=True),
        ScenarioConfig(name="Volume Discount 10%", volume_discount=0.10),
        ScenarioConfig(name="Volume Discount 20%", volume_discount=0.20),
        ScenarioConfig(name="Router Infra $0.10/1M", infra_usd_per_1m=0.10),
        ScenarioConfig(
            name="Worst-Case Combined Stress Test",
            gemini_38_price="list_2027",
            deepseek_window="peak",
            claude_cache_ttl="1h",
            long_context=True,
            infra_usd_per_1m=0.08,
            bl_video_sec_per_clip=8.0,
        ),
    ]

    from sim.config import active_catalog_models
    from sim.monte_carlo import CohortSampler, evaluate_cohorts

    # Common Random Numbers (CRN): sample cohort behaviors once and reuse identically across all scenarios
    base_sc = ScenarioConfig(name="Baseline")
    sampler = CohortSampler(n_cohorts=1000, cohort_size=100, seed=123)
    all_models = active_catalog_models()
    behaviors = sampler.sample_behaviors(all_models, base_sc)
    results = []

    for sc in scenarios:
        for pack in packs:
            if pack.price_bdt <= 0:
                continue
            _, summ = evaluate_cohorts(pack, behaviors, all_models, scenario=sc, n_cohorts=1000, cohort_size=100)
            results.append(
                {
                    "scenario": sc.name,
                    "platform": pack.platform,
                    "pack": pack.name,
                    "mean_contribution_bdt": summ["contribution_bdt_mean"],
                    "gross_margin_mean": summ["gross_margin_mean"],
                    "gross_margin_p05": summ["gross_margin_p05"],
                    "profitable_cohort_share": summ["profitable_cohort_share"],
                    "mean_unprofitable_user_share": summ["mean_unprofitable_user_share"],
                }
            )

    return pd.DataFrame(results)

