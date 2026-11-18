"""Plain-language executive charts for all three telcos (Banglalink, Robi, Grameenphone).

Designed for business decision-makers with zero AI jargon:
- Transparent backgrounds on all charts with safe bounding boxes.
- Complete mathematical reconciliation across exhaustive grids, vulnerability corners, and Monte Carlo.
- Unified executive question-answered chart titles and consistent footers.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

from sim.config import (
    FX_BUY,
    INFRA_USD_PER_1M,
    banglalink_packs,
    gp_packs,
    token_packs,
)
from sim.exhaustive import _text_cogs_bdt, gated_grid_profitability, scan_rates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
FONT_DIR = os.path.join(ROOT, "fonts")

# Brand color palette
COMPANY = {
    "banglalink": "#F26422",  # Banglalink orange
    "robi": "#EC008C",        # Robi magenta
    "gp": "#00AEEF",          # Grameenphone cyan-blue
}
COMPANY_TEXT = {
    "banglalink": "#C24E16",
    "robi": "#A80072",
    "gp": "#0077A8",
}
COMPANY_NAME = {
    "banglalink": "Banglalink",
    "robi": "Robi",
    "gp": "Grameenphone",
}
INK = "#1a1a1a"
MUTED = "#5c6570"
ZERO = "#333333"
PALE = "#E6E8EB"
GP_CHAT_TOKENS = 2_000_000

OLD_FIGURES = (
    "heatmap_contribution.png",
    "heatmap_profitable_share.png",
    "heatmap_gp_pro_media.png",
    "distribution_contribution.png",
    "distribution_cdf.png",
)


def _font(weight):
    return font_manager.FontProperties(fname=os.path.join(FONT_DIR, f"Poppins-{weight}.ttf"))


def _style():
    for weight in ("Regular", "Medium", "SemiBold"):
        font_path = os.path.join(FONT_DIR, f"Poppins-{weight}.ttf")
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Poppins",
            "font.size": 11,
            "axes.unicode_minus": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _per_100(contribution, price):
    if price <= 0:
        return np.nan
    return 100.0 * contribution / price


def _taka(amount):
    amount = float(amount)
    verb = "keeps" if amount >= 0 else "loses"
    return f"{verb} Tk {abs(amount):,.0f}"


def _signed_tk(amount):
    amount = float(amount)
    if round(amount) < 0:
        return f"-Tk {abs(amount):,.0f}"
    return f"Tk {amount:,.0f}"


def _signed_tk_1d(amount):
    amount = float(amount)
    if round(amount, 1) < 0:
        return f"-Tk {abs(amount):,.1f}"
    return f"Tk {amount:,.1f}"


def _fmt_count(n):
    n = float(n)
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n):,}"


def _usage_row(usage, platform, pack, utilization, image_fraction):
    part = usage[
        (usage["platform"] == platform)
        & (usage["pack"] == pack)
        & np.isclose(usage["utilization"], utilization)
        & np.isclose(usage["image_fraction"], image_fraction)
    ]
    return part.iloc[0]


def _crosses_zero(utils, values, cap_tokens):
    """Tokens where a falling line crosses zero. None if it never does."""
    utils = np.asarray(utils, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.all(values > 0):
        return None
    if values[0] <= 0:
        return 0.0
    idx = int(np.argmax(values <= 0))
    left, right = idx - 1, idx
    span = values[left] - values[right]
    frac = 0.0 if span == 0 else values[left] / span
    util = utils[left] + (utils[right] - utils[left]) * frac
    return float(util * cap_tokens)


def _chat_rows(usage):
    rows = []
    for pack in token_packs():
        if pack.price_bdt <= 0 or pack.tokens is None:
            continue
        part = usage[
            (usage["platform"] == pack.platform)
            & (usage["pack"] == pack.name)
            & np.isclose(usage["image_fraction"], 0.0)
        ].sort_values("utilization")
        if len(part) == 0:
            continue
        full = part.iloc[-1]
        pictures = _usage_row(usage, pack.platform, pack.name, 1.0, 1.0)
        crossed = _crosses_zero(part["utilization"], part["contribution_p50"], pack.tokens)
        rows.append(
            {
                "platform": pack.platform,
                "pack": pack.name,
                "price": pack.price_bdt,
                "cap": pack.tokens,
                "crossed": crossed,
                "full_p50": float(full["contribution_p50"]),
                "full_min": float(full["contribution_min"]),
                "full_share": float(full["profitable_share"]),
                "picture_p50": float(pictures["contribution_p50"]),
                "picture_cost": float(pictures["image_bdt"]),
                "net": float(pictures["net_bdt"]),
            }
        )
    return rows


def _gp_contribution(row, tokens, rate_per_1m):
    cost = _text_cogs_bdt(tokens, rate_per_1m / 1_000_000.0)
    return float(row["room_bdt"]) - cost


def _gated_facts(gated_df, pack_list, scans_cache=None):
    scans = scans_cache if scans_cache is not None else {}
    facts = []
    for pack in pack_list:
        if pack.price_bdt <= 0:
            continue
        part = gated_df[(gated_df["platform"] == pack.platform) & (gated_df["pack"] == pack.name)]
        if len(part) == 0:
            continue
        aud_col = part["audio_fraction"] if "audio_fraction" in part.columns else pd.Series(0.0, index=part.index)
        bare_match = part[
            np.isclose(part["image_fraction"], 0.0)
            & np.isclose(part["video_fraction"], 0.0)
            & np.isclose(aud_col, 0.0)
        ]
        bare = bare_match.iloc[0] if len(bare_match) > 0 else part.iloc[0]

        vid_target = 1.0 if pack.video_sec > 0 else 0.0
        aud_target = 1.0 if pack.audio > 0 else 0.0
        full_match = part[
            np.isclose(part["image_fraction"], 1.0)
            & np.isclose(part["video_fraction"], vid_target)
            & np.isclose(aud_col, aud_target)
        ]
        full = full_match.iloc[-1] if len(full_match) > 0 else part.iloc[-1]

        side_path = os.path.join(OUT, "exhaustive_vs_realistic_side_by_side.csv")
        side_df = pd.read_csv(side_path) if os.path.exists(side_path) else None
        full_share = None
        total_scens = None
        if side_df is not None:
            s_match = side_df[(side_df["platform"] == pack.platform) & (side_df["pack"] == pack.name)]
            if len(s_match) > 0 and pd.notna(s_match.iloc[0]["ex_profitable_share"]) and pd.notna(s_match.iloc[0]["ex_combinations"]):
                full_share = float(s_match.iloc[0]["ex_profitable_share"])
                total_scens = int(s_match.iloc[0]["ex_combinations"])

        if full_share is None or total_scens is None:
            key = tuple(pack.models)
            if key not in scans:
                scans[key] = scan_rates(pack.models)
            scan = scans[key]
            # Full multi-axis grid profitability across (media x file x 0.5M-5.5M chat_volume x text_case)
            full_share, total_scens = gated_grid_profitability(part, scan)
            bare_share, bare_scens = gated_grid_profitability(part.iloc[[0]], scan, chat_tokens=GP_CHAT_TOKENS)
        else:
            bare_share, bare_scens = np.nan, 0

        facts.append(
            {
                "pack": pack,
                "platform": pack.platform,
                "bare": bare,
                "full": full,
                "share": full_share,
                "total_scenarios": total_scens,
                "bare_share": bare_share,
                "bare_scenarios": bare_scens,
                "chat_p50": _gp_contribution(bare, GP_CHAT_TOKENS, bare["usd_per_1m_p50"]),
                "chat_min": _gp_contribution(bare, GP_CHAT_TOKENS, bare["usd_per_1m_max"]),
                "media_p50": _gp_contribution(full, GP_CHAT_TOKENS, bare["usd_per_1m_p50"]),
            }
        )
    return facts


def _finish(fig, path, footer):
    import gc
    fig.text(0.04, 0.02, footer, fontsize=8.6, color=MUTED, fontproperties=_font("Regular"), va="bottom")
    fig.savefig(path, dpi=220, facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)
    gc.collect()
    return path


def _named(platform, pack):
    return f"{COMPANY_NAME[platform]} · {pack}"


def _grouped_positions(groups, pitch=1.0):
    positions = []
    labels = []
    platforms = []
    gaps = []
    current_y = 0.0
    for g_idx, (platform, items) in enumerate(groups):
        if g_idx > 0:
            gaps.append(current_y - 0.5 * pitch)
        for item in items:
            positions.append(current_y)
            labels.append(_named(platform, item["label"]))
            platforms.append(platform)
            current_y += pitch
        current_y += 0.8 * pitch
    return positions, labels, platforms, gaps


def _color_ticks(ax, platforms):
    ticks = ax.get_yticklabels()
    for tick, plat in zip(ticks, platforms):
        tick.set_color(COMPANY_TEXT[plat])
        tick.set_fontproperties(_font("Medium"))


def _bare_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d0d3d6")
    ax.spines["bottom"].set_color("#d0d3d6")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#f0f2f5", lw=0.9, zorder=0)


def _title(fig, text, subtitle):
    fig.text(0.04, 0.96, text, fontsize=12.2, color=INK, fontproperties=_font("SemiBold"), va="top")
    fig.text(0.04, 0.922, subtitle, fontsize=9.2, color=MUTED, fontproperties=_font("Regular"), va="top")


# -----------------------------------------------------------------------------
# CHART 1: When the fee runs out
# -----------------------------------------------------------------------------
def _fee_chart(chat, bl_facts, gp_facts):
    robi_items = []
    for row in chat:
        if row["crossed"] is None:
            covered = 1.0
            sentence = f"Allowance runs out first · Company stays profitable ({_taka(row['full_p50'])})"
        else:
            covered = row["crossed"] / row["cap"]
            sentence = f"Fee runs out at {covered * 100:.0f}% of allowance · Remaining chats lose money"
        robi_items.append({"label": row["pack"], "covered": covered, "sentence": sentence})

    robi_groups = [("robi", robi_items)]
    positions, labels, platforms, gaps = _grouped_positions(robi_groups, pitch=1.1)

    fig = plt.figure(figsize=(14.8, 12.4))
    # Leave ample vertical room below main title/subtitle so top subplot title never overlaps
    top = fig.add_axes((0.20, 0.60, 0.76, 0.21))
    bottom = fig.add_axes((0.20, 0.095, 0.76, 0.415))

    for y, platform, item in zip(positions, platforms, robi_items):
        color = COMPANY[platform]
        top.barh(y, 1.0, color=PALE, height=0.62, zorder=2)
        top.barh(y, item["covered"], color=color, height=0.62, zorder=3)
        top.text(1.03, y, item["sentence"], va="center", ha="left", fontsize=9.2, color=COMPANY_TEXT[platform], fontproperties=_font("Medium"))
    top.set_yticks(positions, labels)
    _color_ticks(top, platforms)
    top.set_ylim(positions[-1] + 0.7, -0.6)
    top.set_xlim(0, 2.55)
    top.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    top.set_xticklabels(["0%", "25%", "50%", "75%", "100% cap"])
    top.set_xlabel("Share of customer's prepaid chat allowance used (0% unused to 100% full allowance)")
    _bare_axes(top)
    top.set_title(
        "Robi Prepaid Token Packs: Full colored bar = 100% commercially safe. Gray = Money runs out before allowance.",
        loc="left",
        fontsize=10.3,
        color=INK,
        pad=9,
        fontproperties=_font("Medium"),
    )

    # Bottom axes: Flat Subscriptions (Banglalink + GP)
    gated_groups = []
    bl_items = []
    for fact in bl_facts:
        be = float(fact["bare"]["break_even_p50"]) / 1_000_000.0
        bl_items.append({"label": fact["pack"].name, "length": be})
    gated_groups.append(("banglalink", bl_items))

    gp_items = []
    for fact in gp_facts:
        be = float(fact["bare"]["break_even_p50"]) / 1_000_000.0
        gp_items.append({"label": fact["pack"].name, "length": be})
    gated_groups.append(("gp", gp_items))

    g_pos, g_labels, g_platforms, g_gaps = _grouped_positions(gated_groups, pitch=1.1)
    flat_gated = [item for _, grp in gated_groups for item in grp]

    for y, plat, item in zip(g_pos, g_platforms, flat_gated):
        bottom.barh(y, item["length"], color=COMPANY[plat], height=0.62, zorder=2)
        bottom.text(
            item["length"] + 0.12,
            y,
            f"{item['length']:.1f} million chat tokens, then additional chats lose money",
            va="center",
            ha="left",
            fontsize=9.2,
            color=COMPANY_TEXT[plat],
            fontproperties=_font("Medium"),
        )
    bottom.set_yticks(g_pos, g_labels)
    _color_ticks(bottom, g_platforms)
    bottom.set_ylim(g_pos[-1] + 0.7, -0.6)
    bottom.set_xlim(0, 17.5)
    bottom.set_xlabel("Text-only chat volume (Million Tokens) until flat fee is spent, before deducting free media quotas.")
    _bare_axes(bottom)
    for gap in g_gaps:
        bottom.axhline(gap, color="#e6e8eb", lw=1)
    bottom.set_title(
        "Flat Subscriptions (Banglalink & GP): Unlimited chat with no token cap (text-only break-even before media quotas).",
        loc="left",
        fontsize=10.3,
        color=INK,
        pad=9,
        fontproperties=_font("Medium"),
    )

    _title(
        fig,
        "If a customer uses their subscription, does their allowance run out first or does the company lose money first?",
        "Top: Robi prepaid packs have token caps. Full colored bar = Allowance runs out first. Gray = Money runs out before allowance.\n"
        "Bottom: Banglalink & GP have unlimited chat (no token caps); bars show how much text chat the flat fee covers before flipping to a loss.",
    )
    safe_pairs = [(it, r) for it, r in zip(robi_items, chat) if r["crossed"] is None]
    crossed_pairs = [(it, r) for it, r in zip(robi_items, chat) if r["crossed"] is not None]
    safe_r_txt = ", ".join(it["label"] for it, _ in safe_pairs) if safe_pairs else "None"
    safe_range = f"+Tk {min(r['full_p50'] for _, r in safe_pairs):,.0f} to +Tk {max(r['full_p50'] for _, r in safe_pairs):,.0f}" if safe_pairs else "n/a"
    crossed_txt = ", ".join(f"{it['label']} ({it['covered'] * 100:.0f}% of {_fmt_count(r['cap'])})" for it, r in crossed_pairs) if crossed_pairs else "none"
    be_min_m = min(it["length"] for it in flat_gated)
    be_max_m = max(it["length"] for it in flat_gated)
    return _finish(
        fig,
        os.path.join(FIG, "when_the_fee_runs_out.png"),
        f"Layman takeaway: On Robi ({safe_r_txt}), even using 100% of tokens on typical median chat stays profitable ({safe_range}); breaks even first on {crossed_txt}.\n"
        f"Banglalink and Grameenphone offer unlimited chat with no token cap to stop heavy users, so the bars mark the exact text volume ({be_min_m:.1f}M to {be_max_m:.1f}M tokens) where profit turns negative.",
    )


# -----------------------------------------------------------------------------
# CHART 2: What pictures and media quotas add (Single Unified Chart)
# -----------------------------------------------------------------------------
def _picture_chart(chat, bl_facts, gp_facts):
    groups = []

    bl_group = []
    for fact in bl_facts:
        price = fact["pack"].price_bdt
        bl_group.append(
            {
                "label": fact["pack"].name,
                "chat": _per_100(fact["chat_p50"], price),
                "pictures": _per_100(fact["media_p50"], price),
            }
        )
    groups.append(("banglalink", bl_group))

    robi_group = []
    for row in chat:
        robi_group.append(
            {
                "label": row["pack"],
                "chat": _per_100(row["full_p50"], row["price"]),
                "pictures": _per_100(row["picture_p50"], row["price"]),
            }
        )
    groups.append(("robi", robi_group))

    gp_group = []
    for fact in gp_facts:
        price = fact["pack"].price_bdt
        gp_group.append(
            {
                "label": fact["pack"].name,
                "chat": _per_100(fact["chat_p50"], price),
                "pictures": _per_100(fact["media_p50"], price),
            }
        )
    groups.append(("gp", gp_group))

    positions, labels, platforms, gaps = _grouped_positions(groups, pitch=1.5)
    flat = [item for _, group in groups for item in group]
    fig_h = 2.4 + 0.52 * len(positions)
    fig, ax = plt.subplots(figsize=(15.8, fig_h))

    robi_pic_val = abs(robi_group[0]["pictures"]) if robi_group else 0.0
    ax.axvline(0, color=ZERO, lw=1.2, ls="--", zorder=1)
    for y, platform, item in zip(positions, platforms, flat):
        color = COMPANY[platform]
        ax.plot([item["pictures"], item["chat"]], [y, y], color=color, lw=3.0, zorder=2)
        ax.scatter(item["chat"], y, s=80, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.scatter(item["pictures"], y, s=80, facecolors="white", edgecolors=color, linewidths=2.5, zorder=3)

        c_str = _signed_tk(item["chat"])
        if platform == "robi":
            p_str = _signed_tk_1d(item["pictures"])
            diff_text = f"Chat: {c_str}  |  Media/Files: {p_str} (Capped at ~{abs(item['pictures']):.1f}% VAT + BTRC wedge via token debit)"
        else:
            p_str = _signed_tk(item["pictures"])
            if item["pictures"] < 0:
                diff_text = f"Chat: {c_str}  |  Media: {p_str} loss (Un-debited quota risk!)"
            else:
                diff_text = f"Chat: {c_str}  |  Media: {p_str} (Media within fee)"

        max_pt = max(item["chat"], item["pictures"])
        ax.text(max_pt + 6.0, y, diff_text, va="center", ha="left", fontsize=9.1, color=COMPANY_TEXT[platform], fontproperties=_font("Medium"))

    ax.set_yticks(positions, labels)
    _color_ticks(ax, platforms)
    ax.set_ylim(positions[-1] + 0.85, -0.6)
    min_x = min(-420.0, min(item["pictures"] for item in flat) - 70.0)
    max_x = max(340.0, max(item["chat"] for item in flat) + 275.0)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel("Profit kept by the telco from each Tk 100 paid by the customer. Left of 0 = Company loses money.")
    _bare_axes(ax)
    for gap in gaps:
        ax.axhline(gap, color="#e6e8eb", lw=1)

    _title(
        fig,
        "How much profit does the telco keep per Tk 100 paid when a customer generates pictures and media?",
        "Solid dot: Customer uses 100% of allowance (or 2M baseline chat) on text only.\n"
        "Open ring: Customer uses 100% of bundled image/video/audio quota (or converts Robi tokens into pictures/files).",
    )
    fig.tight_layout(rect=(0, 0.08, 0.96, 0.88))
    worst_media_pct = min(item["pictures"] for item in flat)
    return _finish(
        fig,
        os.path.join(FIG, "what_pictures_add.png"),
        f"Layman takeaway: Robi deducts pictures and files directly from prepaid tokens, capping worst-case media exposure identically across packs at -Tk {robi_pic_val:.1f} per Tk 100 (15% app VAT + 6.5% BTRC/SOF wedge).\n"
        f"Banglalink (Pro, Creator, Legend) and Grameenphone bundle large media quotas into flat fees; full quota usage triggers wholesale losses up to -Tk {abs(worst_media_pct):,.0f} per Tk 100 paid.",
    )


# -----------------------------------------------------------------------------
# CHART 3: How often it makes money (Full Deterministic Grid across ALL combinations)
# -----------------------------------------------------------------------------
def _often_chart(chat, bl_facts, gp_facts):
    groups = []

    bl_items = []
    for fact in bl_facts:
        bl_items.append(
            {
                "label": fact["pack"].name,
                "share": fact["share"] * 100.0,
                "scenarios": f"{_fmt_count(fact['total_scenarios'])} full",
            }
        )
    groups.append(("banglalink", bl_items))

    ex_summary_path = os.path.join(OUT, "exhaustive_summary.csv")
    ex_sum = pd.read_csv(ex_summary_path) if os.path.exists(ex_summary_path) else None
    robi_items = []
    for row in chat:
        sh = row["full_share"] * 100.0
        scen = "142.5M full"
        if ex_sum is not None:
            match = ex_sum[(ex_sum["platform"] == "robi") & (ex_sum["pack"] == row["pack"])]
            if len(match) > 0:
                sh = float(match.iloc[0]["profitable_share"]) * 100.0
                scen = f"{_fmt_count(match.iloc[0]['combinations'])} full"
        robi_items.append({"label": row["pack"], "share": sh, "scenarios": scen})
    groups.append(("robi", robi_items))

    gp_items = []
    for fact in gp_facts:
        gp_items.append(
            {
                "label": fact["pack"].name,
                "share": fact["share"] * 100.0,
                "scenarios": f"{_fmt_count(fact['total_scenarios'])} full",
            }
        )
    groups.append(("gp", gp_items))

    positions, labels, platforms, gaps = _grouped_positions(groups, pitch=1.5)
    flat = [item for _, group in groups for item in group]
    fig_h = 2.8 + 0.52 * len(positions)
    fig, ax = plt.subplots(figsize=(14.8, fig_h))

    for y, platform, item in zip(positions, platforms, flat):
        ax.barh(y, item["share"], color=COMPANY[platform], height=0.60, zorder=2)
        if item["share"] >= 75.0:
            status = "Commercially Safe"
        elif item["share"] >= 50.0:
            status = "Moderate Risk"
        else:
            status = "High Loss Risk"
        label_text = f"{item['share']:.1f}% profitable  ·  {status} ({item['scenarios']} scenarios tested)"
        ax.text(item["share"] + 1.8, y, label_text, va="center", ha="left", fontsize=9.2, color=COMPANY_TEXT[platform], fontproperties=_font("Medium"))

    ax.axvline(50, color=ZERO, lw=1.2, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(positions, labels)
    _color_ticks(ax, platforms)
    ax.set_ylim(positions[-1] + 0.75, -0.6)
    ax.set_xlim(0, 185)
    ax.set_xlabel("Share of tested market + media + usage scenarios where the pack is profitable (%)")
    _bare_axes(ax)
    for gap in gaps:
        ax.axhline(gap, color="#e6e8eb", lw=1)

    bl_min_s, bl_max_s = min(f["total_scenarios"] for f in bl_facts), max(f["total_scenarios"] for f in bl_facts)
    gp_min_s, gp_max_s = min(f["total_scenarios"] for f in gp_facts), max(f["total_scenarios"] for f in gp_facts)
    robi_scens = robi_items[0]["scenarios"].replace(" full", "") if robi_items else "142.5M"
    _title(
        fig,
        "If a single user exhibits ANY possible behavior, what percentage of theoretical scenarios are profitable?",
        "Exhaustive grid across ALL model mixes, prompt lengths, cache hit rates, media/file quotas, AND 0.5M–5.5M chat volumes (or 0–100% pool on Robi).\n"
        f"Banglalink tests {_fmt_count(bl_min_s)}–{_fmt_count(bl_max_s)} scenarios, Robi {robi_scens} scenarios, and GP {_fmt_count(gp_min_s)}–{_fmt_count(gp_max_s)} scenarios. Dashed line = 50% break-even mark.",
    )
    fig.tight_layout(rect=(0, 0.08, 0.96, 0.86))
    robi_min_sh, robi_max_sh = min(i["share"] for i in robi_items), max(i["share"] for i in robi_items)
    bl_basic_adv = [i["share"] for i in bl_items if "Basic" in i["label"] or "Advanced" in i["label"]]
    bl_cr = next((i["share"] for i in bl_items if "Creator" in i["label"]), 0.0)
    bl_lg = next((i["share"] for i in bl_items if "Legend" in i["label"]), 0.0)
    gp_min_sh, gp_max_sh = min(i["share"] for i in gp_items), max(i["share"] for i in gp_items)
    return _finish(
        fig,
        os.path.join(FIG, "how_often_it_makes_money.png"),
        "Layman takeaway: Every bar sweeps ALL model mixes, prompt ratios, cache hits, media/file quotas, and chat volumes.\n"
        f"Robi packs ({robi_min_sh:.0f}%–{robi_max_sh:.0f}%) and Banglalink Basic/Advanced ({min(bl_basic_adv):.0f}%–{max(bl_basic_adv):.0f}%) stay profitable via token debit or tiny media caps.\n"
        f"Large un-debited media quotas erase profitability once usage scales: Banglalink Creator ({bl_cr:.1f}%), Legend ({bl_lg:.1f}%), and GP ({gp_min_sh:.1f}%–{gp_max_sh:.1f}%).",
    )


# -----------------------------------------------------------------------------
# CHART 4: Picture & media bill relative to fee
# -----------------------------------------------------------------------------
def _picture_bill_chart(chat, bl_facts, gp_facts):
    items = {"banglalink": [], "robi": [], "gp": []}

    for fact in bl_facts:
        net = float(fact["full"]["net_bdt"])
        media = net - float(fact["full"]["room_bdt"])
        name = fact["pack"].name
        if fact["pack"].video_sec > 0 and fact["pack"].audio > 0:
            name += " + video & audio"
        elif fact["pack"].video_sec > 0:
            name += " + video"
        items["banglalink"].append({"label": name, "times": media / net})

    for row in chat:
        items["robi"].append({"label": row["pack"], "times": row["picture_cost"] / row["net"]})

    for fact in gp_facts:
        net = float(fact["full"]["net_bdt"])
        media = net - float(fact["full"]["room_bdt"])
        name = fact["pack"].name
        if fact["pack"].video_sec > 0 and fact["pack"].audio > 0:
            name += " + video & audio"
        elif fact["pack"].video_sec > 0:
            name += " + video"
        items["gp"].append({"label": name, "times": media / net})

    groups = [(name, items[name]) for name in ("banglalink", "robi", "gp")]
    positions, labels, platforms, gaps = _grouped_positions(groups, pitch=1.5)
    flat = [item for _, group in groups for item in group]
    fig_h = 2.6 + 0.54 * len(positions)
    fig, ax = plt.subplots(figsize=(14.5, fig_h))

    ax.axvline(1.0, color="#666666", lw=1.2, ls="--", alpha=0.55, zorder=1)
    ax.text(1.03, -0.58, "Entire Subscription Fee (1.0×)", va="bottom", ha="left", fontsize=9.2, color=ZERO, fontproperties=_font("SemiBold"))

    for y, platform, item in zip(positions, platforms, flat):
        ax.barh(y, item["times"], color=COMPANY[platform], height=0.60, zorder=2)
        if platform == "robi":
            detail = f"{item['times']:.2f}× fee (Deducted from token pool · Capped)"
        elif item["times"] > 1.0:
            detail = f"{item['times']:.2f}× fee (Free quota exceeds fee · Loss!)"
        else:
            detail = f"{item['times']:.2f}× fee (Media within subscription fee)"
        ax.text(
            item["times"] + 0.08,
            y,
            detail,
            va="center",
            ha="left",
            fontsize=9.2,
            color=COMPANY_TEXT[platform],
            fontproperties=_font("Medium"),
            zorder=4,
        )

    ax.set_yticks(positions, labels)
    _color_ticks(ax, platforms)
    ax.set_ylim(positions[-1] + 0.85, -1.15)
    max_t = max(item["times"] for item in flat)
    ax.set_xlim(0, max(9.2, max_t + 4.4))
    ax.set_xlabel("Wholesale cost of generating included images, video & audio, divided by net revenue kept by the telco")
    _bare_axes(ax)
    for gap in gaps:
        ax.axhline(gap, color="#e6e8eb", lw=1)

    _title(
        fig,
        "How many times larger than the customer's subscription fee is the wholesale cost of their included pictures?",
        "The dashed line at 1.0× represents the entire net fee the company keeps from the customer. Past that line, included media alone loses money.",
    )
    fig.tight_layout(rect=(0, 0.12, 0.96, 0.88))
    excess_times = [it["times"] for p in ("banglalink", "gp") for it in items[p] if it["times"] > 1.0]
    min_ex, max_ex = (min(excess_times), max(excess_times)) if excess_times else (0.0, 0.0)
    robi_times = items["robi"][0]["times"] if items["robi"] else 0.0
    return _finish(
        fig,
        os.path.join(FIG, "pictures_cost_more_than_the_fee.png"),
        "Layman takeaway: The vertical line at 1.0× marks 100% of net revenue kept after statutory BTRC/SOF (6.5%) and app-inclusive 15% VAT on Banglalink/Robi.\n"
        f"• Robi: Pictures debit prepaid tokens, capping wholesale exposure identically at {robi_times:.2f}× net revenue regardless of plan tier.\n"
        f"• Un-debited quotas (Banglalink Pro/Creator/Legend & GP tiers exceeding 1.0×): maxing out included media costs {min_ex:.2f}× to {max_ex:.2f}× net revenue before text chat.",
    )


# -----------------------------------------------------------------------------
# CHART 5: Everyday customer vs. Worst-case power user (Unified with MC & Corners)
# -----------------------------------------------------------------------------
def _priciest_chart(mc_summary_path, corners_path):
    mc_df = pd.read_csv(mc_summary_path)
    corners_df = pd.read_csv(corners_path)

    items = {"banglalink": [], "robi": [], "gp": []}
    for platform in ("banglalink", "robi", "gp"):
        p_mc = mc_df[(mc_df["platform"] == platform) & (mc_df["revenue_bdt_mean"] > 0)]
        for _, mrow in p_mc.iterrows():
            pack_name = str(mrow["pack"])
            c_match = corners_df[(corners_df["platform"] == platform) & (corners_df["pack"] == pack_name)]
            if len(c_match) == 0:
                continue
            crow = c_match.iloc[0]
            usual_val = float(mrow["contribution_bdt_mean"]) / float(mrow.get("users_per_cohort", 100))
            worst_val = float(crow["contribution_bdt"])
            model_name = str(crow["dearest_model"])
            items[platform].append(
                {
                    "label": pack_name,
                    "usual": usual_val,
                    "worst": worst_val,
                    "model": model_name,
                }
            )

    groups = [(name, items[name]) for name in ("banglalink", "robi", "gp") if items[name]]
    positions, labels, platforms, gaps = _grouped_positions(groups, pitch=2.2)
    flat = [item for _, group in groups for item in group]

    fig_h = 3.8 + 0.56 * len(positions)
    fig, ax = plt.subplots(figsize=(15.8, fig_h))

    for y, platform, item in zip(positions, platforms, flat):
        color = COMPANY[platform]
        worst_val = item["worst"]
        usual_val = item["usual"]

        # 1. Worst-case vulnerability corner bar
        ax.barh(y - 0.28, worst_val, left=0, height=0.34, color=color, alpha=0.85, zorder=2)
        if worst_val < 0:
            ax.text(
                worst_val - 120,
                y - 0.28,
                f"-Tk {abs(worst_val):,.0f} loss ({item['model']})",
                va="center",
                ha="right",
                fontsize=8.6,
                color=COMPANY_TEXT[platform],
                fontproperties=_font("Medium"),
                zorder=4,
            )
        else:
            ax.text(
                worst_val + 100,
                y - 0.28,
                f"+Tk {worst_val:,.0f} profit ({item['model']}, worst-case safe)",
                va="center",
                ha="left",
                fontsize=8.6,
                color=COMPANY_TEXT[platform],
                fontproperties=_font("SemiBold"),
                zorder=4,
            )

        # 2. Everyday customer average (hatched bar to the right)
        ax.barh(y + 0.28, usual_val, left=0, height=0.34, facecolor=color, alpha=0.30, hatch="///", edgecolor=color, linewidth=1.5, zorder=2)
        if usual_val >= 0:
            ax.text(
                usual_val + 100,
                y + 0.28,
                f"+Tk {usual_val:,.0f} profit (everyday average)",
                va="center",
                ha="left",
                fontsize=8.6,
                color=COMPANY_TEXT[platform],
                fontproperties=_font("Medium"),
                zorder=4,
            )
        else:
            # Put small negative everyday loss just right of Tk 0 when a huge negative corner bar sits left
            if usual_val > -500.0 and worst_val < -1000.0:
                ax.text(
                    100.0,
                    y + 0.28,
                    f"-Tk {abs(usual_val):,.0f} loss (everyday average)",
                    va="center",
                    ha="left",
                    fontsize=8.6,
                    color=COMPANY_TEXT[platform],
                    fontproperties=_font("SemiBold"),
                    zorder=4,
                )
            else:
                ax.text(
                    worst_val - 120 if worst_val < usual_val else usual_val - 120,
                    y + 0.28,
                    f"-Tk {abs(usual_val):,.0f} loss (everyday average)",
                    va="center",
                    ha="right",
                    fontsize=8.6,
                    color=COMPANY_TEXT[platform],
                    fontproperties=_font("SemiBold"),
                    zorder=4,
                )

    ax.axvline(0, color=ZERO, lw=1.6, ls="--", zorder=3)
    ax.text(0, -0.82, "Tk 0 (Break-even)", ha="center", va="bottom", fontsize=9.8, color=INK, fontproperties=_font("SemiBold"))

    ax.set_yticks(positions, labels)
    _color_ticks(ax, platforms)
    ax.set_ylim(positions[-1] + 1.0, -1.35)

    x_lo = min(-7800.0, min(i["worst"] for i in flat) - 2700.0)
    x_hi = max(2200.0, max(i["usual"] for i in flat) + 1700.0)
    ax.set_xlim(x_lo, x_hi)
    xticks = [t for t in (-8000, -6000, -4000, -2000, 0, 1000) if x_lo + 400 <= t <= x_hi - 300]
    ax.set_xticks(xticks)
    ax.set_xticklabels([("Tk 0" if t == 0 else (f"+Tk {t:,}" if t > 0 else f"-Tk {abs(t):,}")) for t in xticks])
    ax.set_xlabel("Net Profit or Loss per Subscriber on a Number Line: Left of Tk 0 = Loss · Right of Tk 0 = Profit")
    _bare_axes(ax)

    for gap in gaps:
        ax.axhline(gap, color="#e6e8eb", lw=1)

    legend_elements = [
        Patch(facecolor="#5c6570", alpha=0.85, label="Worst-Case Corner: 100% priciest model, 0% cache, 3:1 I:O, full media (+ BL files)"),
        Patch(facecolor="#5c6570", alpha=0.30, hatch="///", edgecolor="#5c6570", label="Everyday Customer: Realistic cohort average (Monte Carlo mean)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, frameon=False, prop=_font("Medium"))

    _title(
        fig,
        "What is the profit on an everyday customer vs. the worst-case loss if a user maxes out the priciest AI model?",
        "Reconciles Monte Carlo everyday cohort averages against Vulnerability Corner stress bounds. Right of Tk 0 = Profit · Left of Tk 0 = Loss.",
    )
    fig.tight_layout(rect=(0, 0.08, 0.96, 0.84))
    min_u = min(i["usual"] for i in flat)
    max_u = max(i["usual"] for i in flat)
    bl_worst_it = min(items["banglalink"], key=lambda x: x["worst"])
    robi_worst_it = min(items["robi"], key=lambda x: x["worst"])
    gp_worst_it = min(items["gp"], key=lambda x: x["worst"])
    safe_corners = [
        f"{COMPANY_NAME[plat]} {it['label']} (+Tk {it['worst']:,.0f})"
        for plat in ("banglalink", "robi", "gp")
        for it in items[plat]
        if it["worst"] >= 0
    ]
    safe_txt = ("only " + ", ".join(safe_corners) + " stays profitable") if safe_corners else "every pack turns unprofitable"
    return _finish(
        fig,
        os.path.join(FIG, "priciest_model_vs_usual_mix.png"),
        f"Layman takeaway: On an average real-world customer (hatched bars), packs range from {_signed_tk(min_u)} to {_signed_tk(max_u)} per user.\n"
        f"In a worst-case vulnerability corner (100% priciest AI model, 0% cache, 3:1 output-heavy chat, full media + Banglalink files), {safe_txt}.\n"
        f"Heavy media/flagship tiers plunge up to -Tk {abs(bl_worst_it['worst']):,.0f} (Banglalink {bl_worst_it['label']}), -Tk {abs(robi_worst_it['worst']):,.0f} (Robi {robi_worst_it['label']}), and -Tk {abs(gp_worst_it['worst']):,.0f} (GP {gp_worst_it['label']}).",
    )


# -----------------------------------------------------------------------------
# CHART 6: Wholesale Supplier Cost Avoidance (27-Model Active MC Distribution)
# -----------------------------------------------------------------------------
def _menu_chart(mc_summary_or_df):
    df = pd.read_csv(mc_summary_or_df) if isinstance(mc_summary_or_df, (str, os.PathLike)) else mc_summary_or_df
    bl_order = ["Basic Tk 100", "Advanced Tk 300", "Pro Tk 500", "Creator Tk 1,000", "Legend Tk 1,500"]
    gp_order = ["Student Tk 71", "Lite Tk 99", "Plus Tk 299", "Pro Tk 799"]

    groups = []
    for platform, order in (("banglalink", bl_order), ("gp", gp_order)):
        sub = df[(df["platform"] == platform) & (df["revenue_bdt_mean"] > 0)].copy()
        if len(sub) == 0:
            continue
        by_pack = sub.set_index("pack")
        items = []
        for name in order:
            if name not in by_pack.index:
                continue
            r = by_pack.loc[name]
            avoided_usd = float(r["gating_avoided_usd_per_1m_mean"])
            cheaper_sh = float(r["gating_share_whitelist_cheaper"]) * 100.0
            items.append({"label": name, "share": cheaper_sh, "saved": avoided_usd * FX_BUY})
        if items:
            groups.append((platform, items))

    positions, labels, platforms, gaps = _grouped_positions(groups, pitch=1.5)
    flat = [item for _, grp in groups for item in grp]

    fig_h = 2.4 + 0.54 * len(positions)
    fig, ax = plt.subplots(figsize=(15.2, fig_h))

    for y, plat, item in zip(positions, platforms, flat):
        color = COMPANY[plat]
        share = item["share"]
        taka = item["saved"]
        ax.barh(y, share, color=color, height=0.62, zorder=2)
        note = f"Avoids ~Tk {taka:,.0f} in wholesale costs per 1M tokens  ·  {share:.0f}% of 27-model demand cheaper"
        ax.text(share + 1.2, y, note, va="center", ha="left", fontsize=9.2, color=COMPANY_TEXT[plat], fontproperties=_font("Medium"))

    ax.axvline(50, color=ZERO, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(positions, labels)
    _color_ticks(ax, platforms)
    ax.set_ylim(positions[-1] + 0.8, -0.6)
    ax.set_xlim(0, 215)
    ax.set_xlabel("Share of 1,000,000 simulated user mixes (all 27 active AI models) where gating cuts wholesale supplier costs (%)")
    _bare_axes(ax)

    for gap in gaps:
        ax.axhline(gap, color="#e6e8eb", lw=1)

    _title(
        fig,
        "How much wholesale supplier cost do telcos avoid by locking budget packs to lightweight AI models?",
        "Evaluated over all 27 active AI models in the 1,000,000-user Monte Carlo simulation. Restricting menus cuts supplier bills so flat fees survive.",
    )
    fig.tight_layout(rect=(0, 0.10, 0.96, 0.88))
    budget_saved = [it["saved"] for it in flat if not any(k in it["label"] for k in ("Creator", "Legend", "Pro Tk 799"))]
    prem_saved = [it["saved"] for it in flat if any(k in it["label"] for k in ("Creator", "Legend", "Pro Tk 799"))]
    return _finish(
        fig,
        os.path.join(FIG, "blocking_models.png"),
        "Layman takeaway: Evaluated across 1,000,000 simulated users over the full 27-model market catalog (deterministic 15-model uniform grid also stored in gating_breakdown.csv).\n"
        f"By locking entry/mid tiers strictly to budget models, telcos save Tk {min(budget_saved):,.0f}–{max(budget_saved):,.0f} per 1M chat tokens served.\n"
        f"On premium flagship tiers (Banglalink Creator/Legend, GP Pro), savings drop to Tk {min(prem_saved):,.0f}–{max(prem_saved):,.0f} per 1M tokens.",
    )


# -----------------------------------------------------------------------------
# CHART 7: Continuous Monte Carlo Cohort Margins
# -----------------------------------------------------------------------------
def _monte_carlo_chart(mc_summary_path):
    if not os.path.exists(mc_summary_path):
        return None
    df = pd.read_csv(mc_summary_path)
    df = df.dropna(subset=["gross_margin_mean"]).copy()
    df = df[df["revenue_bdt_mean"] > 0].copy()
    if len(df) == 0:
        return None

    groups = []
    for platform in ("banglalink", "robi", "gp"):
        sub = df[df["platform"] == platform]
        items = []
        for _, row in sub.iterrows():
            items.append({
                "label": str(row["pack"]),
                "platform": platform,
                "p05": float(row["gross_margin_p05"]) * 100.0,
                "p50": float(row["gross_margin_p50"]) * 100.0,
                "p95": float(row["gross_margin_p95"]) * 100.0,
                "mean": float(row["gross_margin_mean"]) * 100.0,
                "profitable_share": float(row["profitable_cohort_share"]) * 100.0,
            })
        if items:
            groups.append((platform, items))

    positions, labels, platforms, gaps = _grouped_positions(groups, pitch=1.6)
    flat = [item for _, group in groups for item in group]
    fig_h = 2.4 + 0.48 * (positions[-1] + 1)
    fig, ax = plt.subplots(figsize=(14.8, fig_h))

    for y, platform, item in zip(positions, platforms, flat):
        color = COMPANY[platform]
        ax.plot([item["p05"], item["p95"]], [y, y], color=color, lw=5.0, zorder=2, solid_capstyle="round")
        ax.scatter(item["p50"], y, color=color, s=120, zorder=3, edgecolors="white", linewidth=2.0)
        label_text = f"Median {item['p50']:.1f}%  [Range: {item['p05']:.1f}% to {item['p95']:.1f}%]  ·  {item['profitable_share']:.0f}% of cohorts profitable"
        ax.text(item["p95"] + 3.0, y, label_text, va="center", ha="left", fontsize=9.2, color=COMPANY_TEXT[platform], fontproperties=_font("Medium"))

    ax.axvline(0, color=ZERO, lw=1.5, ls="--", zorder=2)
    ax.text(0.8, positions[-1] + 0.55, "Break-even (0%)", fontsize=8.5, color=MUTED, fontproperties=_font("Regular"))
    ax.set_yticks(positions, labels)
    _color_ticks(ax, platforms)
    ax.set_ylim(positions[-1] + 0.85, -0.6)

    min_p05_item = min(flat, key=lambda x: x["p05"])
    max_p95_item = max(flat, key=lambda x: x["p95"])
    x_lo = min(-25.0, (np.floor(min_p05_item["p05"] / 10.0) - 1.0) * 10.0)
    ax.set_xlim(x_lo, 180)
    xticks = [t for t in (-100, -80, -60, -40, -20, 0, 25, 50, 75, 100) if t >= x_lo]
    ax.set_xticks(xticks)
    ax.set_xlabel("Subscriber Gross Margin % (Showing 90% Confidence Interval: 5th percentile to 95th percentile, with Median Dot)")
    _bare_axes(ax)

    for gap in gaps:
        ax.axhline(gap, color="#e6e8eb", lw=1)

    robi_rows = [i for plat, grp in groups if plat == "robi" for i in grp]
    robi_med_min = min(i["p50"] for i in robi_rows)
    robi_med_max = max(i["p50"] for i in robi_rows)
    robi_prof_min = min(i["profitable_share"] for i in robi_rows)
    at_risk = [i for i in flat if i["profitable_share"] < 99.9]
    risk_txt = (
        ", ".join(f"{COMPANY_NAME[i['platform']]} {i['label']} ({i['profitable_share']:.0f}% profitable)" for i in at_risk)
        if at_risk
        else "All packs reach ~100% profitable cohorts under realistic usage"
    )

    _title(
        fig,
        "When 100 real subscribers buy this pack together, what gross margin percentage will the telco actually see?",
        "10,000 simulated cohorts of 100 users (Banglalink/Robi VAT-inclusive; GP USD 0.035–0.050/img & Grok 4.1 Fast on Lite; 4–8s BL video clips & file context).\n"
        f"Unclipped 90% CIs spanning {min_p05_item['p05']:.1f}% ({COMPANY_NAME[min_p05_item['platform']]} {min_p05_item['label']} 5th pct) to +{max_p95_item['p95']:.1f}% ({COMPANY_NAME[max_p95_item['platform']]} {max_p95_item['label']} 95th pct).",
    )
    fig.tight_layout(rect=(0, 0.08, 0.96, 0.86))
    return _finish(
        fig,
        os.path.join(FIG, "monte_carlo_cohort_margins.png"),
        "Layman takeaway: Real cohorts average ~35% light/dormant users, ~50% core users, and ~15% power users.\n"
        f"Robi packs deliver {robi_med_min:.0f}%–{robi_med_max:.0f}% median margins ({robi_prof_min:.0f}%+ cohorts profitable). Cohort loss risk concentrates in: {risk_txt}.",
    )


# -----------------------------------------------------------------------------
# MASTER PLOT RUNNER
# -----------------------------------------------------------------------------
def main():
    _style()
    os.makedirs(FIG, exist_ok=True)
    for name in OLD_FIGURES:
        path = os.path.join(FIG, name)
        if os.path.exists(path):
            os.remove(path)
    usage = pd.read_csv(os.path.join(OUT, "exhaustive_by_usage.csv"))
    gated_df = pd.read_csv(os.path.join(OUT, "gated_by_media.csv"))
    mc_summary_path = os.path.join(OUT, "monte_carlo_summary.csv")
    corners_path = os.path.join(OUT, "vulnerability_corners.csv")

    scans_cache = {}
    chat = _chat_rows(usage)
    bl_facts = _gated_facts(gated_df, banglalink_packs(), scans_cache=scans_cache)
    gp_facts = _gated_facts(gated_df, gp_packs(), scans_cache=scans_cache)

    paths = [
        _fee_chart(chat, bl_facts, gp_facts),
        _picture_chart(chat, bl_facts, gp_facts),
        _often_chart(chat, bl_facts, gp_facts),
        _picture_bill_chart(chat, bl_facts, gp_facts),
        _priciest_chart(mc_summary_path, corners_path),
        _menu_chart(mc_summary_path),
    ]
    mc_chart = _monte_carlo_chart(mc_summary_path)
    if mc_chart:
        paths.append(mc_chart)
    for path in paths:
        print(f"Generated chart: {path}")


if __name__ == "__main__":
    main()
