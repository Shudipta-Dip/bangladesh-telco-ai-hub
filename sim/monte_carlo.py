"""Continuous Cohort Monte Carlo Simulation Engine.

Simulates 10,000 heterogeneous 100-user cohorts sampled from continuous empirical distributions
grounded in Bangladesh mobile data (BTRC, 85/15 modified Pareto, prepaid breakage)
and real-world GenAI token distributions (Lognormal core with Pareto power-user tail).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sim.config import (
    BL_TOKENS_PER_FILE,
    BL_VIDEO_SEC_PER_CLIP,
    BTRC,
    CHURN_REFUND,
    DEFAULT_IMAGE_USD,
    FX_BUY,
    FX_WAR,
    INFRA_USD_PER_1M,
    MDR,
    MODELS,
    OPEN_MODELS,
    ROBI_IMAGE_USD,
    SOF,
    VIDEO_USD_PER_SEC,
    VOLUME_DISCOUNT,
    Pack,
    ScenarioConfig,
    banglalink_tokens_per_image,
    build_models,
    tokens_per_image_for_pack,
)
from sim.engine import effective_video_sec, net_revenue_bdt, recognized_and_cash
from sim.pricing import price_factors


class CohortSampler:
    """Vectorized continuous sampling for N users across M cohorts."""

    def __init__(self, n_cohorts: int = 10_000, cohort_size: int = 100, seed: int = 42):
        self.n_cohorts = n_cohorts
        self.cohort_size = cohort_size
        self.n_total = n_cohorts * cohort_size
        self.rng = np.random.default_rng(seed)

    def sample_behaviors(self, all_models: List[str], scenario: Optional[ScenarioConfig] = None):
        """Draw continuous user characteristics based on empirical segments.

        Returns a dictionary of arrays, each of shape (n_total, ...).
        """
        n = self.n_total
        sc = scenario or ScenarioConfig()

        # 1. Segment assignment: 35% Dormant, 50% Core, 15% Power
        segment_choices = np.array([0, 1, 2], dtype=int)
        segment_probs = np.array([0.35, 0.50, 0.15])
        segments = self.rng.choice(segment_choices, size=n, p=segment_probs)

        # 2. Token Utilization u in [0, 1] via Beta distributions
        utilization = np.zeros(n, dtype=float)
        # Dormant: Beta(1.2, 5.0) -> mean ~19%
        mask0 = segments == 0
        utilization[mask0] = self.rng.beta(1.2, 5.0, size=np.sum(mask0))
        # Core: Beta(3.0, 3.0) -> mean ~50%
        mask1 = segments == 1
        utilization[mask1] = self.rng.beta(3.0, 3.0, size=np.sum(mask1))
        # Power: Beta(6.0, 1.5) -> mean ~80%
        mask2 = segments == 2
        utilization[mask2] = self.rng.beta(6.0, 1.5, size=np.sum(mask2))
        utilization = np.clip(utilization, 0.001, 1.0)

        # 3. Cache Hit Rate h in [0, 1] via Beta distributions
        cache_hit = np.zeros(n, dtype=float)
        cache_hit[mask0] = self.rng.beta(1.5, 4.0, size=np.sum(mask0))  # ~27%
        cache_hit[mask1] = self.rng.beta(3.0, 2.5, size=np.sum(mask1))  # ~55%
        cache_hit[mask2] = self.rng.beta(7.0, 1.5, size=np.sum(mask2))  # ~82%
        cache_hit = np.clip(cache_hit, 0.0, 0.98)

        # 4. Cache Write Share w in [0, 1 - h]
        write_factor = np.zeros(n, dtype=float)
        write_factor[mask0] = self.rng.uniform(0.05, 0.15, size=np.sum(mask0))
        write_factor[mask1] = self.rng.uniform(0.10, 0.25, size=np.sum(mask1))
        write_factor[mask2] = self.rng.uniform(0.15, 0.35, size=np.sum(mask2))
        cache_write = write_factor * (1.0 - cache_hit)

        # 5. Continuous Input:Output Ratio via Log-Mixture
        # Dormant: mode ~ 4:1 (chat)
        # Core: mode ~ 12:1 (context chat / docs)
        # Power: mode ~ 180:1 (large automated loops / multi-file processing)
        io_ratio = np.zeros(n, dtype=float)
        io_ratio[mask0] = np.exp(self.rng.normal(np.log(4.0), 0.35, size=np.sum(mask0)))
        io_ratio[mask1] = np.exp(self.rng.normal(np.log(12.0), 0.50, size=np.sum(mask1)))
        io_ratio[mask2] = np.exp(self.rng.normal(np.log(180.0), 0.60, size=np.sum(mask2)))
        io_ratio = np.clip(io_ratio, 1.0, 350.0)

        # 6. Media fractions (Zero-inflated continuous Beta)
        image_fraction = np.zeros(n, dtype=float)
        # Dormant: 75% zero, 25% Beta(0.8, 4.0)
        dormant_img_active = self.rng.random(size=np.sum(mask0)) < 0.25
        image_fraction[mask0] = np.where(dormant_img_active, self.rng.beta(0.8, 4.0, size=np.sum(mask0)), 0.0)
        # Core: 85% active Beta(2.0, 3.0)
        core_img_active = self.rng.random(size=np.sum(mask1)) < 0.85
        image_fraction[mask1] = np.where(core_img_active, self.rng.beta(2.0, 3.0, size=np.sum(mask1)), 0.0)
        # Power: 95% active Beta(5.0, 2.0)
        power_img_active = self.rng.random(size=np.sum(mask2)) < 0.95
        image_fraction[mask2] = np.where(power_img_active, self.rng.beta(5.0, 2.0, size=np.sum(mask2)), 0.0)
        image_fraction = np.clip(image_fraction, 0.0, 1.0)

        # Video fraction (GP Pro & BL Pro/Creator/Legend)
        video_fraction = np.zeros(n, dtype=float)
        dormant_vid_active = self.rng.random(size=np.sum(mask0)) < 0.15
        video_fraction[mask0] = np.where(dormant_vid_active, self.rng.uniform(0.0, 0.3, size=np.sum(mask0)), 0.0)
        core_vid_active = self.rng.random(size=np.sum(mask1)) < 0.40
        video_fraction[mask1] = np.where(core_vid_active, self.rng.uniform(0.1, 0.6, size=np.sum(mask1)), 0.0)
        power_vid_active = self.rng.random(size=np.sum(mask2)) < 0.75
        video_fraction[mask2] = np.where(power_vid_active, self.rng.beta(4.0, 1.5, size=np.sum(mask2)), 0.0)
        video_fraction = np.clip(video_fraction, 0.0, 1.0)

        # Independent audio fraction (Banglalink Creator/Legend music & voice generation)
        audio_fraction = np.zeros(n, dtype=float)
        dormant_aud_active = self.rng.random(size=np.sum(mask0)) < 0.15
        audio_fraction[mask0] = np.where(dormant_aud_active, self.rng.beta(0.8, 5.0, size=np.sum(mask0)), 0.0)
        core_aud_active = self.rng.random(size=np.sum(mask1)) < 0.45
        audio_fraction[mask1] = np.where(core_aud_active, self.rng.beta(1.5, 4.0, size=np.sum(mask1)), 0.0)
        power_aud_active = self.rng.random(size=np.sum(mask2)) < 0.75
        audio_fraction[mask2] = np.where(power_aud_active, self.rng.beta(3.0, 2.5, size=np.sum(mask2)), 0.0)
        audio_fraction = np.clip(audio_fraction, 0.0, 1.0)

        # Independent file attachment fraction + context tokens per file (Banglalink un-debited files)
        file_fraction = np.zeros(n, dtype=float)
        dormant_file_active = self.rng.random(size=np.sum(mask0)) < 0.40
        file_fraction[mask0] = np.where(dormant_file_active, self.rng.beta(1.0, 5.0, size=np.sum(mask0)), 0.0)
        core_file_active = self.rng.random(size=np.sum(mask1)) < 0.75
        file_fraction[mask1] = np.where(core_file_active, self.rng.beta(2.0, 3.0, size=np.sum(mask1)), 0.0)
        power_file_active = self.rng.random(size=np.sum(mask2)) < 0.95
        file_fraction[mask2] = np.where(power_file_active, self.rng.beta(4.0, 2.0, size=np.sum(mask2)), 0.0)
        file_fraction = np.clip(file_fraction, 0.0, 1.0)
        bl_tokens_per_file = self.rng.uniform(15_000.0, 25_000.0, size=n)

        # GP continuous image unit price draw (Uniform $0.035..$0.050 for packs without published image SKUs)
        gp_image_usd = self.rng.uniform(0.035, 0.050, size=n)

        # Banglalink variable video clip duration draw (Uniform 4.0s..8.0s per clip)
        bl_clip_sec = self.rng.uniform(4.0, 8.0, size=n)

        # 7. Model mix weights via Continuous Dirichlet over model tiers
        models_dict = build_models(sc)
        n_models = len(all_models)
        tiers = [models_dict[m].tier for m in all_models]

        # Prior alphas for economy, mid, flagship per segment
        tier_alphas = {
            0: {"economy": 6.0, "mid": 2.0, "flagship": 0.2},   # Dormant
            1: {"economy": 3.0, "mid": 5.0, "flagship": 1.0},   # Core
            2: {"economy": 0.5, "mid": 2.5, "flagship": 7.0},   # Power
        }

        # Build alpha vector per segment
        alphas_by_seg = np.zeros((3, n_models), dtype=float)
        for s in range(3):
            for i, t in enumerate(tiers):
                alphas_by_seg[s, i] = tier_alphas[s][t]

        # Sample gamma random variables and normalize in-place in chunks
        gammas = np.zeros((n, n_models), dtype=float)
        for s in range(3):
            s_mask = segments == s
            s_count = int(np.sum(s_mask))
            if s_count > 0:
                gammas[s_mask] = self.rng.gamma(alphas_by_seg[s], 1.0, size=(s_count, n_models))

        chunk_norm = 100_000
        for st in range(0, n, chunk_norm):
            ed = min(n, st + chunk_norm)
            gammas[st:ed] /= np.sum(gammas[st:ed], axis=1, keepdims=True)
        model_weights = gammas

        # 8. Continuous Macro FX and TT commission per cohort
        # Drawn once per cohort of 100 users
        cohort_fx = self.rng.normal(sc.fx_buy, 0.60, size=self.n_cohorts)
        cohort_fx = np.clip(cohort_fx, 121.0, 127.0)
        user_fx = np.repeat(cohort_fx, self.cohort_size)

        cohort_tt = self.rng.uniform(0.0, 0.015, size=self.n_cohorts)
        user_tt = np.repeat(cohort_tt, self.cohort_size)

        cohort_infra = self.rng.uniform(0.02, 0.06, size=self.n_cohorts)
        user_infra = np.repeat(cohort_infra, self.cohort_size)

        # Robi 4 Operating Modes (30% Saver, 45% Balanced, 20% Expert, 5% Beast)
        robi_mode = self.rng.choice(
            ["saver", "balanced", "expert", "beast"],
            size=n,
            p=[0.30, 0.45, 0.20, 0.05],
        )

        return {
            "segments": segments,
            "utilization": utilization,
            "cache_hit": cache_hit,
            "cache_write": cache_write,
            "io_ratio": io_ratio,
            "image_fraction": image_fraction,
            "video_fraction": video_fraction,
            "audio_fraction": audio_fraction,
            "file_fraction": file_fraction,
            "bl_tokens_per_file": bl_tokens_per_file,
            "gp_image_usd": gp_image_usd,
            "bl_clip_sec": bl_clip_sec,
            "model_weights": model_weights,
            "robi_mode": robi_mode,
            "fx_buy": user_fx,
            "tt_commission": user_tt,
            "infra_per_1m": user_infra,
        }


def evaluate_cohorts(
    pack: Pack,
    behaviors: Dict[str, np.ndarray],
    all_models: List[str],
    scenario: Optional[ScenarioConfig] = None,
    n_cohorts: int = 10_000,
    cohort_size: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate pack P&L in memory-compact cohort batches and aggregate by cohort."""
    sc = scenario or ScenarioConfig()
    models_dict = build_models(sc)
    n_total = n_cohorts * cohort_size

    miss, read, write, output = price_factors(pack.models, models_dict)
    recognized, _ = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
    net_bdt = float(net_revenue_bdt(recognized, btrc=sc.btrc, sof=sc.sof, mdr=sc.mdr, churn_refund=sc.churn_refund))

    if pack.platform == "robi":
        m_idx = {m: i for i, m in enumerate(pack.models)}

        w_saver = np.zeros(len(pack.models))
        w_saver[m_idx["deepseek-v4-flash"]] = 0.50
        w_saver[m_idx["gemini-2.5-flash-lite"]] = 0.50

        w_balanced = np.zeros(len(pack.models))
        w_balanced[m_idx["deepseek-v4-flash"]] = 0.175
        w_balanced[m_idx["gemini-2.5-flash-lite"]] = 0.175
        w_balanced[m_idx["gpt-5-nano"]] = 0.175
        w_balanced[m_idx["grok-4.1-fast-nr"]] = 0.175
        w_balanced[m_idx["gemini-3.5-flash-lite"]] = 0.10
        w_balanced[m_idx["kimi-k2.6"]] = 0.10
        w_balanced[m_idx["claude-haiku-4.5"]] = 0.05
        w_balanced[m_idx["grok-4.3"]] = 0.05

        w_expert = np.zeros(len(pack.models))
        w_expert[m_idx["deepseek-v4-flash"]] = 0.05
        w_expert[m_idx["gemini-2.5-flash-lite"]] = 0.05
        w_expert[m_idx["gpt-5-nano"]] = 0.05
        w_expert[m_idx["grok-4.1-fast-nr"]] = 0.05
        w_expert[m_idx["gemini-3.5-flash-lite"]] = 0.15
        w_expert[m_idx["kimi-k2.6"]] = 0.15
        w_expert[m_idx["claude-haiku-4.5"]] = 0.25
        w_expert[m_idx["grok-4.3"]] = 0.25

        w_beast = np.zeros(len(pack.models))
        w_beast[m_idx["claude-haiku-4.5"]] = 0.50
        w_beast[m_idx["grok-4.3"]] = 0.50

        mode_matrix = {"saver": w_saver, "balanced": w_balanced, "expert": w_expert, "beast": w_beast}
    else:
        pos_map = {m: i for i, m in enumerate(all_models)}
        allowed_pos = [pos_map[m] for m in pack.models if m in pos_map]
        cheapest_allowed_idx = min(range(len(pack.models)), key=lambda i: (models_dict[pack.models[i]].output, models_dict[pack.models[i]].miss))
        o_miss, o_read, o_write, o_out = price_factors(all_models, models_dict)

    if pack.kind == "token_pool":
        unit_img_price_tp = pack.image_usd if pack.image_usd is not None else (sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd)
        mean_fx_all = float(np.mean(behaviors["fx_buy"] + (sc.fx_buy - FX_BUY)))
        tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=mean_fx_all, image_usd=unit_img_price_tp, scenario=sc)

    c_rev = np.empty(n_cohorts, dtype=float)
    c_cogs_bdt = np.empty(n_cohorts, dtype=float)
    c_cogs_usd = np.empty(n_cohorts, dtype=float)
    c_contrib = np.empty(n_cohorts, dtype=float)
    c_unprof_share = np.empty(n_cohorts, dtype=float)
    c_max_loss = np.empty(n_cohorts, dtype=float)
    c_breakage = np.empty(n_cohorts, dtype=float)

    gating_open_sum = 0.0
    gating_served_sum = 0.0
    gating_avoided_sum = 0.0
    gating_cheaper_cnt = 0

    chunk_cohorts = 1_000
    for c0 in range(0, n_cohorts, chunk_cohorts):
        c1 = min(n_cohorts, c0 + chunk_cohorts)
        u0, u1 = c0 * cohort_size, c1 * cohort_size
        u_len = u1 - u0

        if pack.platform == "robi":
            if pack.category == "day_pack":
                user_modes = np.full(u_len, "beast")
            else:
                raw_modes = behaviors.get("robi_mode", np.full(n_total, "balanced"))[u0:u1]
                allowed = pack.allowed_modes or ["saver", "balanced", "expert", "beast"]
                if "expert" not in allowed and "beast" not in allowed:
                    # Commercial restriction: Pack 1 (p99) & Pack 2 (p149) route strictly to Saver & Balanced
                    user_modes = np.where(np.isin(raw_modes, ["expert", "beast"]), "balanced", raw_modes)
                elif "beast" not in allowed:
                    # Commercial restriction: Pack 3 (p349) routes to Saver, Balanced, Expert
                    user_modes = np.where(raw_modes == "beast", "expert", raw_modes)
                else:
                    user_modes = raw_modes
            served_w = np.zeros((u_len, len(pack.models)), dtype=float)
            for mode_name, w_vec in mode_matrix.items():
                mask = user_modes == mode_name
                if np.any(mask):
                    served_w[mask] = w_vec
            w_miss = served_w @ miss
            w_read = served_w @ read
            w_write = served_w @ write
            w_out = served_w @ output
            del served_w
        else:
            w_allowed = behaviors["model_weights"][u0:u1, allowed_pos]
            missing = np.maximum(0.0, 1.0 - w_allowed.sum(axis=1))
            w_miss = w_allowed @ miss + missing * miss[cheapest_allowed_idx]
            w_read = w_allowed @ read + missing * read[cheapest_allowed_idx]
            w_write = w_allowed @ write + missing * write[cheapest_allowed_idx]
            w_out = w_allowed @ output + missing * output[cheapest_allowed_idx]

        io = behaviors["io_ratio"][u0:u1]
        in_share = io / (io + 1.0)
        out_share = 1.0 / (io + 1.0)
        hit = behaviors["cache_hit"][u0:u1]
        wr = behaviors["cache_write"][u0:u1]
        miss_share = np.maximum(0.0, 1.0 - hit - wr)

        served_in_1m = miss_share * w_miss + hit * w_read + wr * w_write
        blended = in_share * served_in_1m + out_share * w_out
        usd_per_tok = blended / 1_000_000.0
        served_in_usd_per_tok = served_in_1m / 1_000_000.0

        util = behaviors["utilization"][u0:u1]
        img_frac = behaviors["image_fraction"][u0:u1]
        vid_frac = behaviors["video_fraction"][u0:u1]
        fx = behaviors["fx_buy"][u0:u1] + (sc.fx_buy - FX_BUY)
        tt = behaviors["tt_commission"][u0:u1]
        infra = behaviors["infra_per_1m"][u0:u1] + (sc.infra_usd_per_1m - INFRA_USD_PER_1M)

        if pack.kind == "token_pool":
            budget = pack.tokens * util
            images = img_frac * (budget / tokens_per_img)
            image_tokens = images * tokens_per_img
            text_tokens = np.maximum(0.0, budget - image_tokens)

            image_usd = images * unit_img_price_tp
            image_bdt = image_usd * (1.0 - sc.volume_discount) * fx * (1.0 + tt)
            api_usd = text_tokens * usd_per_tok
            infra_usd = text_tokens * (infra / 1_000_000.0)
            text_bdt = (api_usd * (1.0 - sc.volume_discount) + infra_usd) * fx * (1.0 + tt)
            total_cogs_bdt = text_bdt + image_bdt
            breakage = recognized * (1.0 - util)
        else:
            if pack.platform == "gp":
                default_gp_img = pack.image_usd if pack.image_usd is not None else sc.bl_image_usd
                img_price = behaviors["gp_image_usd"][u0:u1] if "gp_image_usd" in behaviors else np.full(u_len, default_gp_img, dtype=float)
            else:
                img_price = pack.image_usd if pack.image_usd is not None else sc.bl_image_usd
            vid_price = pack.video_usd_per_sec if pack.video_usd_per_sec is not None else sc.video_usd_per_sec
            aud_price = pack.audio_usd if pack.audio_usd is not None else 0.012

            if pack.video_clips is not None and pack.video_clips > 0:
                if abs(sc.bl_video_sec_per_clip - BL_VIDEO_SEC_PER_CLIP) > 1e-12:
                    vid_cap = pack.video_clips * float(sc.bl_video_sec_per_clip)
                else:
                    clip_draw = behaviors["bl_clip_sec"][u0:u1] if "bl_clip_sec" in behaviors else sc.bl_video_sec_per_clip
                    vid_cap = pack.video_clips * clip_draw
            else:
                vid_cap = effective_video_sec(pack, sc)

            aud_frac = behaviors["audio_fraction"][u0:u1] if "audio_fraction" in behaviors else img_frac
            images = img_frac * pack.images
            video_sec = vid_frac * vid_cap
            audio_items = aud_frac * pack.audio
            media_usd = images * img_price + video_sec * vid_price + audio_items * aud_price
            media_bdt = media_usd * (1.0 - sc.volume_discount) * fx * (1.0 + tt)

            seg = behaviors["segments"][u0:u1] if "segments" in behaviors else np.ones(u_len, dtype=int)
            # Power users sample higher monthly text volume (5M to 15M tokens)
            chat_tokens = np.where(
                seg == 2,
                sc.gated_power_chat_tokens_min + util * (sc.gated_power_chat_tokens_max - sc.gated_power_chat_tokens_min),
                sc.gated_chat_tokens_base * util,
            )
            api_usd = chat_tokens * usd_per_tok
            infra_usd = chat_tokens * (infra / 1_000_000.0)
            text_bdt = (api_usd * (1.0 - sc.volume_discount) + infra_usd) * fx * (1.0 + tt)

            if pack.platform == "banglalink" and pack.files > 0:
                if abs(sc.bl_tokens_per_file - BL_TOKENS_PER_FILE) > 1e-12:
                    tok_per_file = float(sc.bl_tokens_per_file)
                else:
                    tok_per_file = behaviors["bl_tokens_per_file"][u0:u1] if "bl_tokens_per_file" in behaviors else sc.bl_tokens_per_file
                f_frac = behaviors["file_fraction"][u0:u1] if "file_fraction" in behaviors else 0.0
                file_tokens = f_frac * pack.files * tok_per_file
                file_api_usd = file_tokens * served_in_usd_per_tok
                file_infra_usd = file_tokens * (infra / 1_000_000.0)
                file_bdt = (file_api_usd * (1.0 - sc.volume_discount) + file_infra_usd) * fx * (1.0 + tt)
            else:
                file_bdt = 0.0

            total_cogs_bdt = text_bdt + media_bdt + file_bdt
            breakage = np.zeros(u_len, dtype=float)

            # 27-Model active catalog gating avoidance via allocation-free @ matrix products
            w_all_sub = behaviors["model_weights"][u0:u1]
            open_in_1m = miss_share * (w_all_sub @ o_miss) + hit * (w_all_sub @ o_read) + wr * (w_all_sub @ o_write)
            open_blended = in_share * open_in_1m + out_share * (w_all_sub @ o_out)
            avoided_1m = open_blended - blended
            gating_open_sum += float(np.sum(open_blended))
            gating_served_sum += float(np.sum(blended))
            gating_avoided_sum += float(np.sum(avoided_1m))
            gating_cheaper_cnt += int(np.sum(avoided_1m > 1e-9))

        total_cogs_usd = total_cogs_bdt / fx
        user_contrib = net_bdt - total_cogs_bdt
        n_c_chunk = c1 - c0
        cogs_2d = total_cogs_bdt.reshape((n_c_chunk, cohort_size))
        usd_2d = total_cogs_usd.reshape((n_c_chunk, cohort_size))
        contrib_2d = user_contrib.reshape((n_c_chunk, cohort_size))
        brk_2d = breakage.reshape((n_c_chunk, cohort_size))

        c_rev[c0:c1] = net_bdt * cohort_size
        c_cogs_bdt[c0:c1] = cogs_2d.sum(axis=1)
        c_cogs_usd[c0:c1] = usd_2d.sum(axis=1)
        c_contrib[c0:c1] = contrib_2d.sum(axis=1)
        c_unprof_share[c0:c1] = np.mean(contrib_2d < 0, axis=1)
        c_max_loss[c0:c1] = np.min(contrib_2d, axis=1)
        c_breakage[c0:c1] = brk_2d.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        c_margin = np.where(c_rev > 0, c_contrib / c_rev, np.nan)

    if pack.kind == "token_pool":
        gating_open_mean = gating_served_mean = gating_avoided_mean = gating_share_cheaper = np.nan
    else:
        gating_open_mean = gating_open_sum / n_total
        gating_served_mean = gating_served_sum / n_total
        gating_avoided_mean = gating_avoided_sum / n_total
        gating_share_cheaper = gating_cheaper_cnt / n_total

    cohorts_df = pd.DataFrame(
        {
            "cohort_id": np.arange(n_cohorts),
            "revenue_bdt": c_rev,
            "cogs_bdt": c_cogs_bdt,
            "cogs_usd": c_cogs_usd,
            "contribution_bdt": c_contrib,
            "gross_margin": c_margin,
            "unprofitable_user_share": c_unprof_share,
            "max_power_user_loss_bdt": c_max_loss,
            "breakage_memo_bdt": c_breakage,
        }
    )

    summary = {
        "platform": pack.platform,
        "pack": pack.name,
        "cohorts": n_cohorts,
        "users_per_cohort": cohort_size,
        "revenue_bdt_mean": float(np.mean(c_rev)),
        "cogs_bdt_mean": float(np.mean(c_cogs_bdt)),
        "cogs_bdt_p05": float(np.percentile(c_cogs_bdt, 5)),
        "cogs_bdt_p50": float(np.percentile(c_cogs_bdt, 50)),
        "cogs_bdt_p95": float(np.percentile(c_cogs_bdt, 95)),
        "cogs_bdt_min": float(np.min(c_cogs_bdt)),
        "cogs_bdt_max": float(np.max(c_cogs_bdt)),
        "contribution_bdt_mean": float(np.mean(c_contrib)),
        "contribution_bdt_p05": float(np.percentile(c_contrib, 5)),
        "contribution_bdt_p50": float(np.percentile(c_contrib, 50)),
        "contribution_bdt_p95": float(np.percentile(c_contrib, 95)),
        "contribution_bdt_min": float(np.min(c_contrib)),
        "contribution_bdt_max": float(np.max(c_contrib)),
        "gross_margin_mean": float(np.nanmean(c_margin)),
        "gross_margin_p05": float(np.nanpercentile(c_margin, 5)),
        "gross_margin_p50": float(np.nanpercentile(c_margin, 50)),
        "gross_margin_p95": float(np.nanpercentile(c_margin, 95)),
        "profitable_cohort_share": float(np.mean(c_contrib >= 0)),
        "mean_unprofitable_user_share": float(np.mean(c_unprof_share)),
        "worst_cohort_loss_bdt": float(np.min(c_contrib)),
        "worst_single_user_loss_bdt": float(np.min(c_max_loss)),
        "breakage_memo_bdt_mean": float(np.mean(c_breakage)),
        "gating_open_usd_per_1m_mean": gating_open_mean,
        "gating_served_usd_per_1m_mean": gating_served_mean,
        "gating_avoided_usd_per_1m_mean": gating_avoided_mean,
        "gating_share_whitelist_cheaper": gating_share_cheaper,
    }

    return cohorts_df, summary


def run_monte_carlo_simulation(
    n_cohorts: int = 10_000,
    cohort_size: int = 100,
    scenario: Optional[ScenarioConfig] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full 10,000-cohort simulation across Banglalink, Robi, and GP."""
    import gc
    from sim.config import active_catalog_models, banglalink_packs, gp_packs, token_packs

    sc = scenario or ScenarioConfig()
    sampler = CohortSampler(n_cohorts=n_cohorts, cohort_size=cohort_size, seed=seed)
    all_models = active_catalog_models()

    print(f"Sampling {n_cohorts:,} cohorts of {cohort_size} users ({sampler.n_total:,} total users) from continuous priors...")
    behaviors = sampler.sample_behaviors(all_models, sc)

    all_packs = token_packs() + [p for p in banglalink_packs() if p.price_bdt > 0] + [p for p in gp_packs() if p.price_bdt > 0]
    summary_rows = []
    cohort_frames = []

    for pack in all_packs:
        c_df, summ = evaluate_cohorts(
            pack,
            behaviors,
            all_models,
            scenario=sc,
            n_cohorts=n_cohorts,
            cohort_size=cohort_size,
        )
        c_df["platform"] = pack.platform
        c_df["pack"] = pack.name
        summary_rows.append(summ)
        cohort_frames.append(c_df)

    del behaviors
    gc.collect()
    summary_df = pd.DataFrame(summary_rows)
    all_cohorts_df = pd.concat(cohort_frames, ignore_index=True)
    return summary_df, all_cohorts_df

