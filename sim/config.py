"""Pack catalog, list prices, Bangladesh cost stack, and scenario configurations.

Prices are USD per 1M tokens unless noted. Research date: 6 October 2026.
Defines separate architectures, catalogs, packs, and media debit treatments for:
- Banglalink (RYZE)
- Robi (EveryAI) - including Saver, Balanced, Expert, Beast modes, Day packs, and image/file token pool debit
- Grameenphone (OneAI) - gated whitelist tiers with no text token cap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Macro defaults (Bangladesh Bank & BTRC schedules, 6 October 2026)
FX_BUY = 123.30
FX_WAR = 123.20
BTRC = 0.055
SOF = 0.01
MDR = 0.0
CHURN_REFUND = 0.0
VOLUME_DISCOUNT = 0.0
INFRA_USD_PER_1M = 0.02
DEFAULT_IMAGE_USD = 0.035  # Nano Banana 2 Lite entry benchmark for unspecified packs
BANGLALINK_IMAGE_USD = DEFAULT_IMAGE_USD  # Legacy alias (BL packs specify per-tier image_usd)
IMAGE_USD = DEFAULT_IMAGE_USD
ROBI_IMAGE_USD = 0.0425  # Midpoint between Nano Banana 2 Lite ($0.035) and GPT Image 2.5 ($0.050)
VIDEO_USD_PER_SEC = 0.10
GP_VIDEO_CAP_SEC = 30.0
BL_VIDEO_SEC_PER_CLIP = 5.0
BL_TOKENS_PER_FILE = 20_000.0
BANK_TT_COMMISSION_PCT = 0.0

IO_RATIOS = (3.0, 5.0, 15.0, 154.0, 275.0, 294.0)
FACTOR_STEP = 0.1
WEIGHT_PARTS = 10

# Default generic routed tier mixes
AUTO_TIER = {"economy": 0.70, "mid": 0.20, "flagship": 0.10}
MANUAL_TIER = {"economy": 0.20, "mid": 0.30, "flagship": 0.50}


@dataclass
class ScenarioConfig:
    """All scenario toggles and operational parameters."""
    name: str = "baseline"
    gemini_38_price: str = "intro"          # "intro" or "list_2027"
    deepseek_window: str = "off_peak"       # "off_peak" or "peak"
    claude_cache_ttl: str = "5m"            # "5m" or "1h"
    long_context: bool = False              # prompts > 200k tokens
    volume_discount: float = 0.0            # 0.0 to 0.30
    infra_usd_per_1m: float = 0.02          # 0.02 to 0.10
    bank_tt_commission_pct: float = 0.0     # 0.0 to 0.025
    churn_refund: float = 0.0               # 0.0 to 0.03
    mdr: float = 0.0                        # 0.0 to 0.0185
    vat_mode: str = "default"               # Legacy override ("default" uses per-operator VAT fields below)
    gp_vat_mode: str = "added_on_top"       # GP adds 15% VAT on top of sticker
    bl_vat_mode: str = "inclusive"          # Banglalink app/digital checkout absorbs 15% VAT inside sticker
    robi_vat_mode: str = "inclusive"        # Robi app/MFS checkout absorbs 15% VAT inside sticker
    fx_buy: float = FX_BUY
    fx_war: float = FX_WAR
    btrc: float = BTRC
    sof: float = SOF
    bl_image_usd: float = DEFAULT_IMAGE_USD
    robi_image_usd: float = ROBI_IMAGE_USD
    video_usd_per_sec: float = VIDEO_USD_PER_SEC
    bl_video_sec_per_clip: float = BL_VIDEO_SEC_PER_CLIP
    bl_tokens_per_file: float = BL_TOKENS_PER_FILE

    @property
    def image_usd(self) -> float:
        """Alias for backward compatibility with older callers."""
        return self.bl_image_usd



class Model:
    def __init__(self, model_id, name, miss, output, cache_read=None, cache_write=None, tier=None):
        self.id = model_id
        self.name = name
        self.miss = float(miss)
        self.output = float(output)
        self.cache_read = None if cache_read is None else float(cache_read)
        self.cache_write = None if cache_write is None else float(cache_write)
        if tier is None:
            if self.output < 2.0:
                tier = "economy"
            elif self.output <= 10.0:
                tier = "mid"
            else:
                tier = "flagship"
        self.tier = tier

    @property
    def read_price(self) -> float:
        if self.cache_read is None:
            return self.miss
        return self.cache_read

    @property
    def write_price(self) -> float:
        if self.cache_write is None:
            return self.miss
        return self.cache_write


def build_models(scenario: Optional[ScenarioConfig] = None) -> Dict[str, Model]:
    """Build the master model catalog with scenario overrides applied."""
    sc = scenario or ScenarioConfig()

    # Gemini 3.8 pricing: intro through Dec 2026, list_2027 from 1 Jan 2027
    if sc.gemini_38_price == "list_2027":
        g38_miss, g38_read, g38_out = 1.50, 0.15, 7.50
    else:
        g38_miss, g38_read, g38_out = 0.75, 0.075, 3.75

    # DeepSeek off-peak vs peak window (2x peak during 07-10 & 12-16 BDT)
    if sc.deepseek_window == "peak":
        ds_flash_miss, ds_flash_read, ds_flash_out = 0.30, 0.006, 1.20
        ds_pro_miss, ds_pro_read, ds_pro_out = 1.32, 0.044, 3.96
    else:
        ds_flash_miss, ds_flash_read, ds_flash_out = 0.15, 0.003, 0.60
        ds_pro_miss, ds_pro_read, ds_pro_out = 0.66, 0.022, 1.98

    # Claude 1-hour cache write TTL is 2x base input; 5-minute is 1.25x
    c_write_mult = 2.0 if sc.claude_cache_ttl == "1h" else 1.25

    # Long-context rates for Sol and Grok
    if sc.long_context:
        sol_miss, sol_read, sol_write, sol_out = 8.00, 0.80, 10.00, 30.00
        grok43_miss, grok43_read, grok43_out = 2.50, 0.40, 5.00
        grok46_miss, grok46_read, grok46_out = 4.00, 1.00, 12.00
    else:
        sol_miss, sol_read, sol_write, sol_out = 4.00, 0.40, 5.00, 20.00
        grok43_miss, grok43_read, grok43_out = 1.25, 0.20, 2.50
        grok46_miss, grok46_read, grok46_out = 2.00, 0.50, 6.00

    rows = [
        # --- Core / Universal models ---
        Model("gpt-5-nano", "GPT-5 Nano", 0.05, 0.40, 0.005, None, "economy"),
        Model("gpt-5-mini", "GPT-5 Mini", 0.25, 2.00, 0.025, None, "mid"),
        Model("gpt-5.4", "GPT-5.4", 2.50, 15.00, 0.25, None, "flagship"),
        Model("gpt-5.6-sol", "GPT-5.6 Sol", sol_miss, sol_out, sol_read, sol_write, "flagship"),
        Model("gpt-5.6-terra", "GPT-5.6 Terra", 2.00, 12.00, 0.20, None, "mid"),
        Model("gpt-5.6-luna", "GPT-5.6 Luna", 0.20, 1.20, 0.02, None, "economy"),
        # --- Anthropic ---
        Model("claude-haiku-4.5", "Claude Haiku 4.5", 1.00, 5.00, 0.10, 1.00 * c_write_mult, "economy"),
        Model("claude-sonnet-4", "Claude Sonnet 4", 3.00, 15.00, 0.30, 3.00 * c_write_mult, "flagship"),
        Model("claude-sonnet-5", "Claude Sonnet 5", 2.00, 10.00, 0.20, 2.00 * c_write_mult, "mid"),
        Model("claude-sonnet-5.5", "Claude Sonnet 5.5", 2.00, 10.00, 0.20, 2.00 * c_write_mult, "mid"),
        Model("claude-opus-5", "Claude Opus 5", 5.00, 25.00, 0.50, 5.00 * c_write_mult, "flagship"),
        Model("claude-opus-5.5", "Claude Opus 5.5", 4.00, 20.00, 0.20, 4.00 * c_write_mult, "flagship"),
        # --- Google ---
        Model("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", 0.10, 0.40, 0.010, None, "economy"),
        Model("gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite", 0.30, 2.50, 0.030, None, "mid"),
        Model("gemini-3-flash", "Gemini 3 Flash", 0.50, 3.00, 0.05, None, "mid"),
        Model("gemini-3.8-flash", "Gemini 3.8 Flash", g38_miss, g38_out, g38_read, None, "mid"),
        # --- xAI ---
        Model("grok-4.1-fast-nr", "Grok 4.1 Fast Non-Reasoning", 0.20, 0.50, 0.050, None, "economy"),
        Model("grok-4.3", "Grok 4.3 / 4.1 Fast", grok43_miss, grok43_out, grok43_read, None, "mid"),
        Model("grok-4.6", "Grok 4.6", grok46_miss, grok46_out, grok46_read, None, "mid"),
        # --- DeepSeek ---
        Model("deepseek-v4-flash", "DeepSeek V4 Flash", ds_flash_miss, ds_flash_out, ds_flash_read, None, "economy"),
        Model("deepseek-v4-pro", "DeepSeek V4 Pro", ds_pro_miss, ds_pro_out, ds_pro_read, None, "mid"),
        # --- Moonshot & Alibaba ---
        Model("kimi-k2.6", "Kimi K2.6", 0.95, 4.00, 0.16, None, "mid"),
        Model("qwen3.8-max", "Qwen3.8-Max", 2.00, 6.00, 2.00, None, "mid"),
        # --- Banglalink RYZE & Updated Catalog Models ---
        Model("gpt-6", "GPT 6", 0.15, 0.60, 0.015, None, "economy"),
        Model("gpt-6.1-terra", "GPT 6.1 Terra", 2.00, 12.00, 0.20, None, "flagship"),
        Model("gemini-3.1", "Gemini 3.1", 0.10, 0.40, 0.010, None, "economy"),
        Model("gemini-3.8", "Gemini 3.8", g38_miss, g38_out, g38_read, None, "mid"),
        Model("claude-haiku-5.5", "Claude Haiku 5.5", 1.00, 5.00, 0.10, 1.00 * c_write_mult, "economy"),
        Model("deepseek-v4.1", "DeepSeek V4.1", ds_flash_miss, ds_flash_out, ds_flash_read, None, "economy"),
        Model("perplexity-sonar-pro", "Perplexity Sonar Pro", 3.00, 15.00, 0.50, None, "flagship"),
        Model("mistral-large-2512", "Mistral Large 2512", 2.00, 6.00, 0.50, None, "mid"),
    ]
    return {m.id: m for m in rows}


MODELS = build_models()

# Banglalink Models (Active RYZE text LLMs across tiers)
BANGLALINK_MODELS = [
    "gpt-6",
    "gpt-6.1-terra",
    "gemini-3.1",
    "gemini-3.8",
    "gemini-3.8-flash",
    "claude-haiku-5.5",
    "claude-sonnet-5",
    "grok-4.3",
    "grok-4.6",
    "deepseek-v4.1",
    "deepseek-v4-pro",
    "qwen3.8-max",
    "perplexity-sonar-pro",
    "mistral-large-2512",
]

# Robi Models (Exact 8 text models from brief)
ROBI_TEXT_MODELS = [
    "claude-haiku-4.5",
    "deepseek-v4-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "kimi-k2.6",
    "gpt-5-nano",
    "grok-4.1-fast-nr",
    "grok-4.3",
]

# Robi 4 Operating Modes
ROBI_SAVER_MODELS = ["deepseek-v4-flash", "gemini-2.5-flash-lite"]
ROBI_BALANCED_TIER = {
    "budget": ["deepseek-v4-flash", "gemini-2.5-flash-lite", "gpt-5-nano", "grok-4.1-fast-nr"],
    "mid": ["gemini-3.5-flash-lite", "kimi-k2.6"],
    "frontier": ["claude-haiku-4.5", "grok-4.3"],
}
ROBI_BEAST_MODELS = ["claude-haiku-4.5", "grok-4.3"]

OPEN_MODELS = [
    "gpt-6.1-terra",
    "gpt-5.4",
    "gpt-5.6-sol",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.8-flash",
    "gemini-3.8",
    "gemini-3.1",
    "grok-4.6",
    "grok-4.3",
    "deepseek-v4.1",
    "deepseek-v4-pro",
    "qwen3.8-max",
    "perplexity-sonar-pro",
    "mistral-large-2512",
]


class Pack:
    def __init__(
        self,
        platform: str,
        pack_id: str,
        name: str,
        price_bdt: float,
        tokens: Optional[float] = None,
        images: float = 0.0,
        video_sec: float = 0.0,
        audio: float = 0.0,
        files: float = 0.0,
        models: Optional[List[str]] = None,
        kind: str = "token_pool",
        validity_days: int = 30,
        category: str = "monthly",
        image_usd: Optional[float] = None,
        video_usd_per_sec: Optional[float] = None,
        audio_usd: Optional[float] = None,
        video_clips: Optional[float] = None,
    ):
        self.platform = platform
        self.pack_id = pack_id
        self.name = name
        self.price_bdt = float(price_bdt)
        self.tokens = None if tokens is None else float(tokens)
        self.images = float(images)
        self.video_sec = float(video_sec)
        self.audio = float(audio)
        self.files = float(files)
        self.models = list(models or [])
        self.kind = kind
        self.validity_days = validity_days
        self.category = category
        self.image_usd = image_usd
        self.video_usd_per_sec = video_usd_per_sec
        self.audio_usd = audio_usd
        self.video_clips = None if video_clips is None else float(video_clips)

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.pack_id}"


def banglalink_packs() -> List[Pack]:
    """Banglalink RYZE AI Hub pricing plans (Updated Gated Whitelists)."""
    basic_models = ["gemini-3.1", "gpt-6", "deepseek-v4.1"]
    adv_models = ["grok-4.3", "gemini-3.1", "gpt-6", "deepseek-v4.1"]
    pro_models = ["gpt-6", "gemini-3.8", "claude-haiku-5.5", "grok-4.3", "deepseek-v4.1"]
    creator_models = [
        "gpt-6",
        "claude-haiku-5.5",
        "claude-sonnet-5",
        "grok-4.3",
        "grok-4.6",
        "gemini-3.1",
        "gemini-3.8-flash",
        "deepseek-v4.1",
        "qwen3.8-max",
        "perplexity-sonar-pro",
        "mistral-large-2512",
    ]
    legend_models = [
        "gpt-6.1-terra",
        "claude-sonnet-5",
        "grok-4.6",
        "gemini-3.8-flash",
        "deepseek-v4-pro",
        "qwen3.8-max",
        "perplexity-sonar-pro",
        "mistral-large-2512",
    ]

    return [
        Pack(
            platform="banglalink",
            pack_id="basic",
            name="Basic Tk 100",
            price_bdt=100,
            tokens=None,
            images=5,
            video_sec=0,
            audio=0,
            files=5,
            models=basic_models,
            kind="gated",
            validity_days=30,
            image_usd=0.003,
        ),
        Pack(
            platform="banglalink",
            pack_id="advanced",
            name="Advanced Tk 300",
            price_bdt=300,
            tokens=None,
            images=15,
            video_sec=0,
            audio=0,
            files=15,
            models=adv_models,
            kind="gated",
            validity_days=30,
            image_usd=0.003,
        ),
        Pack(
            platform="banglalink",
            pack_id="pro",
            name="Pro Tk 500",
            price_bdt=500,
            tokens=None,
            images=54,
            video_sec=65.0,  # 13 videos * 5 sec baseline
            audio=0,
            files=25,
            models=pro_models,
            kind="gated",
            validity_days=30,
            image_usd=0.026,
            video_usd_per_sec=0.05,
            video_clips=13.0,
        ),
        Pack(
            platform="banglalink",
            pack_id="creator",
            name="Creator Tk 1,000",
            price_bdt=1000,
            tokens=None,
            images=272,
            video_sec=325.0,  # 65 videos * 5 sec baseline
            audio=389,
            files=30,
            models=creator_models,
            kind="gated",
            validity_days=30,
            image_usd=0.0336,
            video_usd_per_sec=0.0575,
            audio_usd=0.012,
            video_clips=65.0,
        ),
        Pack(
            platform="banglalink",
            pack_id="legend",
            name="Legend Tk 1,500",
            price_bdt=1500,
            tokens=None,
            images=272,
            video_sec=325.0,  # 65 videos * 5 sec baseline
            audio=389,
            files=80,
            models=legend_models,
            kind="gated",
            validity_days=30,
            image_usd=0.034667,
            video_usd_per_sec=0.062,
            audio_usd=0.0126,
            video_clips=65.0,
        ),
    ]


def robi_packs(models: Optional[List[str]] = None) -> List[Pack]:
    model_list = models or ROBI_TEXT_MODELS
    # Monthly packs (token pool with image/file debit)
    monthly_common = dict(platform="robi", models=model_list, kind="token_pool", video_sec=0.0, validity_days=30, category="monthly")
    return [
        # Monthly packs only (Day packs dropped per specification)
        Pack(pack_id="p99", name="Tk 99", price_bdt=99, tokens=1_000_000, images=0, **monthly_common),
        Pack(pack_id="p149", name="Tk 149", price_bdt=149, tokens=2_000_000, images=0, **monthly_common),
        Pack(pack_id="p349", name="Tk 349", price_bdt=349, tokens=5_000_000, images=0, **monthly_common),
        Pack(pack_id="p599", name="Tk 599", price_bdt=599, tokens=10_000_000, images=0, **monthly_common),
        Pack(pack_id="p999", name="Tk 999", price_bdt=999, tokens=20_000_000, images=0, **monthly_common),
    ]


def gp_packs() -> List[Pack]:
    free_models = ["gpt-5-nano", "gemini-3-flash"]
    student = ["gpt-5-nano", "gemini-3-flash"]
    lite = ["deepseek-v4-flash", "gemini-3-flash", "grok-4.1-fast-nr", "gpt-5-nano"]
    plus = [
        "deepseek-v4-flash",
        "gemini-3-flash",
        "grok-4.3",
        "gpt-5-nano",
        "gemini-3.8-flash",
        "gpt-5-mini",
        "gpt-5.4",
        "kimi-k2.6",
    ]
    pro = [
        "claude-sonnet-4",
        "claude-sonnet-5",
        "claude-opus-5",
        "gpt-5.6-sol",
        "grok-4.6",
        "deepseek-v4-pro",
    ]
    return [
        Pack("gp", "free", "Free", 0, tokens=None, images=0, video_sec=0, models=free_models, kind="gated", validity_days=30, image_usd=0.035),
        Pack("gp", "student", "Student Tk 71", 71, tokens=None, images=10, video_sec=0, models=student, kind="gated", validity_days=30, image_usd=0.035),
        Pack("gp", "lite", "Lite Tk 99", 99, tokens=None, images=30, video_sec=0, models=lite, kind="gated", validity_days=30, image_usd=0.035),
        Pack("gp", "plus", "Plus Tk 299", 299, tokens=None, images=50, video_sec=0, models=plus, kind="gated", validity_days=30, image_usd=0.045),
        Pack("gp", "pro", "Pro Tk 799", 799, tokens=None, images=100, video_sec=GP_VIDEO_CAP_SEC, models=pro, kind="gated", validity_days=30, image_usd=0.050),
    ]


def token_packs(models_bl: Optional[List[str]] = None, models_robi: Optional[List[str]] = None) -> List[Pack]:
    """Token pool packs (Robi only, as Banglalink transitioned to gated subscriptions)."""
    return robi_packs(models_robi)


def gated_packs() -> List[Pack]:
    """Gated whitelist subscription packs (Banglalink + Grameenphone)."""
    return banglalink_packs() + gp_packs()


def active_catalog_models() -> List[str]:
    """Union of all text LLM IDs offered in active commercial packs or OPEN_MODELS."""
    seen = {}
    for pack in robi_packs() + banglalink_packs() + gp_packs():
        for model_id in pack.models:
            seen[model_id] = None
    for model_id in OPEN_MODELS:
        seen[model_id] = None
    return list(seen.keys())


def entry_banglalink_pack() -> Pack:
    return next(p for p in banglalink_packs() if p.pack_id == "basic")


def entry_robi_pack() -> Pack:
    return next(p for p in robi_packs() if p.pack_id == "p99")


def tokens_per_image_for_pack(pack: Pack, fx_buy: float = FX_BUY, image_usd: Optional[float] = None) -> float:
    """Calculate token deduction per image for token pool packs (Robi).

    Deducts tokens from the pool using the pack's implied retail rate:
        tokens_per_image = (image_usd * fx_buy) / (price_bdt / tokens)
    Falls back to the entry token-pool pack (Robi Tk 99) if a gated/free pack lacks a token cap.
    """
    if image_usd is None:
        if pack.image_usd is not None:
            image_usd = pack.image_usd
        else:
            image_usd = ROBI_IMAGE_USD if pack.platform == "robi" else DEFAULT_IMAGE_USD

    if pack.price_bdt <= 0 or pack.tokens is None or pack.tokens <= 0:
        entry = entry_robi_pack()
        retail_bdt_per_token = entry.price_bdt / entry.tokens
    else:
        retail_bdt_per_token = pack.price_bdt / pack.tokens

    return (image_usd * fx_buy) / retail_bdt_per_token


def banglalink_tokens_per_image(pack: Pack, fx_buy: float = FX_BUY, image_usd: float = BANGLALINK_IMAGE_USD) -> float:
    return tokens_per_image_for_pack(pack, fx_buy=fx_buy, image_usd=image_usd)


def robi_tokens_per_image(pack: Pack, fx_buy: float = FX_BUY, image_usd: float = ROBI_IMAGE_USD) -> float:
    return tokens_per_image_for_pack(pack, fx_buy=fx_buy, image_usd=image_usd)
