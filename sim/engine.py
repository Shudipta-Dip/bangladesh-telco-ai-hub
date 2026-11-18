"""Token-pool P&L for Banglalink and Robi, and the separate gated GP P&L.

Fully supports scenario configs, cache-write pricing, VAT treatments, bank TT commission,
pack-specific image/file token pool debit for both Banglalink and Robi, and defensible GP break-even accounting.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from sim.config import (
    AUTO_TIER,
    BANGLALINK_IMAGE_USD,
    BTRC,
    CHURN_REFUND,
    FX_BUY,
    FX_WAR,
    INFRA_USD_PER_1M,
    MANUAL_TIER,
    MDR,
    MODELS,
    OPEN_MODELS,
    ROBI_IMAGE_USD,
    SOF,
    VIDEO_USD_PER_SEC,
    VOLUME_DISCOUNT,
    Pack,
    ScenarioConfig,
    build_models,
    tokens_per_image_for_pack,
)
from sim.pricing import (
    blend_weights,
    one_hot,
    price_factors,
    text_cogs_bdt,
    tier_mix,
    usd_per_input_token,
    usd_per_token,
)


def effective_video_sec(
    pack: Pack,
    scenario: Optional[ScenarioConfig] = None,
    sec_per_clip: Optional[float] = None,
) -> float:
    """Resolve monthly video seconds using clip duration for clip-based packs (Banglalink)."""
    sc = scenario or ScenarioConfig()
    if pack.video_clips is not None and pack.video_clips > 0:
        clip_sec = float(sc.bl_video_sec_per_clip if sec_per_clip is None else sec_per_clip)
        return float(pack.video_clips) * clip_sec
    return float(pack.video_sec)


def net_keep_factor(
    btrc: float = BTRC,
    sof: float = SOF,
    mdr: float = MDR,
    churn_refund: float = CHURN_REFUND,
) -> float:
    return (1.0 - btrc - sof) * (1.0 - mdr) * (1.0 - churn_refund)


def net_revenue_bdt(
    price_bdt,
    btrc: float = BTRC,
    sof: float = SOF,
    mdr: float = MDR,
    churn_refund: float = CHURN_REFUND,
):
    keep = net_keep_factor(btrc, sof, mdr, churn_refund)
    return np.asarray(price_bdt, dtype=float) * keep


def recognized_and_cash(pack: Pack, vat_mode: str = "default", scenario: Optional[ScenarioConfig] = None):
    """Calculate recognized revenue and customer cash collected per operator VAT regime.

    - Explicit `vat_mode != "default"` overrides per-operator ScenarioConfig defaults.
    - "added_on_top": Cash is sticker * 1.15, recognized revenue is sticker (GP default).
    - "inclusive": Cash is sticker, recognized revenue is sticker / 1.15 (BL/Robi app default).
    - "mixed_50_50": 50% inclusive / 50% sticker_is_revenue blend.
    - "sticker_is_revenue": Both cash and recognized revenue equal sticker.
    """
    sc = scenario or ScenarioConfig()
    sticker = float(pack.price_bdt)
    if vat_mode != "default":
        eff_mode = vat_mode
    elif pack.platform == "gp":
        eff_mode = sc.gp_vat_mode
    elif pack.platform == "banglalink":
        eff_mode = sc.bl_vat_mode
    else:
        eff_mode = sc.robi_vat_mode

    if eff_mode == "added_on_top":
        return sticker, sticker * 1.15
    if eff_mode == "inclusive":
        return sticker / 1.15, sticker
    if eff_mode == "mixed_50_50":
        return 0.5 * sticker + 0.5 * (sticker / 1.15), sticker
    return sticker, sticker


class MixBook:
    """Precomputed mixes for one catalog under a scenario."""

    def __init__(self, model_ids, scenario: Optional[ScenarioConfig] = None):
        self.scenario = scenario or ScenarioConfig()
        self.models_dict = build_models(self.scenario)
        self.model_ids = list(model_ids)
        self.manual = tier_mix(self.model_ids, MANUAL_TIER, self.models_dict)
        self.auto = tier_mix(self.model_ids, AUTO_TIER, self.models_dict)
        self.routed_keys, _ = blend_weights(self.manual, self.auto, 0.0)
        self.miss, self.read, self.write, self.output = price_factors(self.routed_keys, self.models_dict)
        self.cheapest = one_hot(self.model_ids, "cheapest", self.models_dict)
        self.dearest = one_hot(self.model_ids, "dearest", self.models_dict)
        self.dearest_id = next(iter(self.dearest))
        self.cheapest_id = next(iter(self.cheapest))

    def weights(self, scenario_name: str, manual_share):
        if scenario_name == "routed":
            keys, weights = blend_weights(self.manual, self.auto, manual_share)
            return keys, weights
        if scenario_name == "all_cheapest":
            return self._fixed(self.cheapest, manual_share)
        if scenario_name == "all_dearest":
            return self._fixed(self.dearest, manual_share)
        raise ValueError(scenario_name)

    def _fixed(self, mapping, manual_share):
        share = np.asarray(manual_share, dtype=float)
        keys = list(mapping.keys())
        vector = np.array([mapping[key] for key in keys], dtype=float)
        weights = np.broadcast_to(vector, share.shape + (len(keys),)).copy()
        return keys, weights

    def dearest_blend(self, dearest_weight):
        weight = np.asarray(dearest_weight, dtype=float)
        if self.dearest_id == self.cheapest_id:
            keys = [self.dearest_id]
            weights = np.ones(weight.shape + (1,))
            return keys, weights
        keys = [self.dearest_id, self.cheapest_id]
        dear = weight[..., None]
        cheap = (1.0 - weight)[..., None]
        return keys, np.concatenate([dear, cheap], axis=-1)


def token_pool_result(
    pack: Pack,
    scenario_name: str,
    utilization,
    cache_hit,
    manual_share,
    io_ratio,
    image_fraction,
    tokens_per_image: Optional[float] = None,
    cache_write_share=0.0,
    scenario: Optional[ScenarioConfig] = None,
):
    """Monthly result for one token-pool pack. Inputs are arrays of the same shape."""
    sc = scenario or ScenarioConfig()
    book = MixBook(pack.models, sc)
    keys, weights = book.weights(scenario_name, manual_share)
    miss, read, write, output = price_factors(keys, book.models_dict)

    per_token = usd_per_token(
        weights,
        io_ratio,
        cache_hit,
        miss,
        read,
        output,
        write=write,
        cache_write_share=cache_write_share,
    )

    utilization = np.asarray(utilization, dtype=float)
    image_fraction = np.asarray(image_fraction, dtype=float)
    budget = pack.tokens * utilization

    # Both Banglalink AND Robi deduct images/files from the token pool!
    unit_img_price = sc.robi_image_usd if pack.platform == "robi" else sc.bl_image_usd
    if tokens_per_image is None:
        tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=sc.fx_buy, image_usd=unit_img_price)
    else:
        tokens_per_img = tokens_per_image

    images = image_fraction * (budget / tokens_per_img)
    image_tokens = images * tokens_per_img
    text_tokens = np.maximum(0.0, budget - image_tokens)
    unused_images = np.zeros_like(budget)

    api_usd, infra_usd, text_bdt = text_cogs_bdt(
        text_tokens,
        per_token,
        volume_discount=sc.volume_discount,
        infra_usd_per_1m=sc.infra_usd_per_1m,
        fx_buy=sc.fx_buy,
        bank_tt_commission_pct=sc.bank_tt_commission_pct,
    )
    image_usd = images * unit_img_price
    image_bdt = image_usd * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
    cogs_bdt = text_bdt + image_bdt
    api_usd = api_usd + image_usd

    recognized, cash = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
    rev = np.full(np.shape(utilization), recognized, dtype=float)
    cash_arr = np.full(np.shape(utilization), cash, dtype=float)
    net = net_revenue_bdt(rev, btrc=sc.btrc, sof=sc.sof, mdr=sc.mdr, churn_refund=sc.churn_refund)
    contribution = net - cogs_bdt
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(net > 0, contribution / net, np.nan)
    breakage = rev * (1.0 - utilization)

    return {
        "cash_bdt": cash_arr,
        "recognized_bdt": rev,
        "net_bdt": net,
        "net_usd": net / sc.fx_war,
        "text_tokens": text_tokens,
        "images": images,
        "unused_images": unused_images,
        "api_usd": api_usd,
        "infra_usd": infra_usd,
        "cogs_bdt": cogs_bdt,
        "cogs_usd": cogs_bdt / sc.fx_buy,
        "contribution_bdt": contribution,
        "gross_margin": margin,
        "breakage_memo_bdt": breakage,
        "image_tokens": image_tokens,
    }


def gp_result(
    pack: Pack,
    scenario_name: str,
    cache_hit,
    manual_share,
    io_ratio,
    image_fraction,
    video_fraction=0.0,
    audio_fraction=0.0,
    file_fraction=0.0,
    sec_per_clip: Optional[float] = None,
    cache_write_share=0.0,
    scenario: Optional[ScenarioConfig] = None,
):
    """Flat-fee gated subscription pack (Banglalink & GP). No text token cap. Text cost is per generated token.

    Gating savings compare this whitelist with the same scenario on the open catalog.
    """
    sc = scenario or ScenarioConfig()
    gated = MixBook(pack.models, sc)
    open_book = MixBook(OPEN_MODELS, sc)
    g_keys, g_weights = gated.weights(scenario_name, manual_share)
    o_keys, o_weights = open_book.weights(scenario_name, manual_share)
    g_miss, g_read, g_write, g_out = price_factors(g_keys, gated.models_dict)
    o_miss, o_read, o_write, o_out = price_factors(o_keys, open_book.models_dict)

    gated_per = usd_per_token(g_weights, io_ratio, cache_hit, g_miss, g_read, g_out, write=g_write, cache_write_share=cache_write_share)
    open_per = usd_per_token(o_weights, io_ratio, cache_hit, o_miss, o_read, o_out, write=o_write, cache_write_share=cache_write_share)
    gated_in_per = usd_per_input_token(g_weights, cache_hit, g_miss, g_read, write=g_write, cache_write_share=cache_write_share)

    image_fraction = np.asarray(image_fraction, dtype=float)
    video_fraction = np.asarray(video_fraction, dtype=float)
    audio_fraction = np.asarray(audio_fraction, dtype=float)
    file_fraction = np.asarray(file_fraction, dtype=float)

    img_price = pack.image_usd if pack.image_usd is not None else sc.bl_image_usd
    vid_price = pack.video_usd_per_sec if pack.video_usd_per_sec is not None else sc.video_usd_per_sec
    aud_price = pack.audio_usd if pack.audio_usd is not None else 0.012
    vid_cap = effective_video_sec(pack, sc, sec_per_clip=sec_per_clip)

    images = image_fraction * pack.images
    video_sec = video_fraction * vid_cap
    audio_items = audio_fraction * pack.audio
    media_usd = images * img_price + video_sec * vid_price + audio_items * aud_price
    media_bdt = media_usd * (1.0 - sc.volume_discount) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)

    # Un-debited file context input tokens on Banglalink only (Robi already debits files from token pool)
    file_tokens = file_fraction * pack.files * sc.bl_tokens_per_file if (pack.platform == "banglalink" and pack.files > 0) else np.zeros_like(file_fraction)
    file_api_usd = file_tokens * gated_in_per
    file_infra_usd = file_tokens * (sc.infra_usd_per_1m / 1_000_000.0)
    file_bdt = (file_api_usd * (1.0 - sc.volume_discount) + file_infra_usd) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)

    recognized, cash = recognized_and_cash(pack, vat_mode=sc.vat_mode, scenario=sc)
    shape = np.shape(np.broadcast_arrays(cache_hit, manual_share, io_ratio, image_fraction, video_fraction, audio_fraction, file_fraction)[0])
    rev = np.full(shape, recognized, dtype=float)
    cash_arr = np.full(shape, cash, dtype=float)
    net = net_revenue_bdt(rev, btrc=sc.btrc, sof=sc.sof, mdr=sc.mdr, churn_refund=sc.churn_refund)
    room_after_media_only = net - media_bdt
    room_bdt = net - media_bdt - file_bdt

    gated_bdt_per_token = (gated_per * (1.0 - sc.volume_discount) + sc.infra_usd_per_1m / 1_000_000.0) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)
    open_bdt_per_token = (open_per * (1.0 - sc.volume_discount) + sc.infra_usd_per_1m / 1_000_000.0) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct)

    # When room_bdt <= 0, media (+ files) alone exceeds net revenue. Token break-even is 0.0 tokens.
    with np.errstate(divide="ignore", invalid="ignore"):
        break_even_tokens = np.where(room_bdt <= 0, 0.0, np.where(gated_bdt_per_token > 0, room_bdt / gated_bdt_per_token, np.nan))

    return {
        "cash_bdt": cash_arr,
        "recognized_bdt": rev,
        "net_bdt": net,
        "net_usd": net / sc.fx_war,
        "images": images,
        "video_sec": video_sec,
        "audio": audio_items,
        "file_tokens": file_tokens,
        "media_usd": media_usd,
        "media_bdt": media_bdt,
        "file_bdt": file_bdt,
        "room_after_media_only_bdt": room_after_media_only,
        "room_after_media_bdt": room_bdt,
        "media_already_unprofitable": np.where(room_bdt <= 0, 1.0, 0.0),
        "gated_usd_per_1m": gated_per * 1_000_000.0,
        "open_usd_per_1m": open_per * 1_000_000.0,
        "gating_avoided_usd_per_1m": (open_per - gated_per) * 1_000_000.0,
        "break_even_tokens": break_even_tokens,
        "dearest_id": gated.dearest_id,
        "cheapest_id": gated.cheapest_id,
    }


gated_result = gp_result


def contribution_at_volume(
    gp_row_usd_per_token,
    tokens,
    net_bdt,
    media_bdt,
    scenario: Optional[ScenarioConfig] = None,
):
    sc = scenario or ScenarioConfig()
    per = gp_row_usd_per_token
    api_usd = tokens * per
    infra_usd = tokens * (sc.infra_usd_per_1m / 1_000_000.0)
    cogs_bdt = (api_usd * (1.0 - sc.volume_discount) + infra_usd) * sc.fx_buy * (1.0 + sc.bank_tt_commission_pct) + media_bdt
    return net_bdt - cogs_bdt


def default_tokens_per_image(pack: Optional[Pack] = None, fx_buy: float = FX_BUY, image_usd: Optional[float] = None) -> float:
    if pack is None:
        from sim.config import entry_robi_pack
        pack = entry_robi_pack()
    return tokens_per_image_for_pack(pack, fx_buy=fx_buy, image_usd=image_usd)
