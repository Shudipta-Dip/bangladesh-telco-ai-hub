"""Unified Side-by-Side Comparison Generator.

Merges and compares core findings across:
1. Deterministic Exhaustive Factor Grid (all 14.3 billion combinations under uniform prior)
2. Empirical Continuous Monte Carlo Simulation (10,000 cohorts x 100 users under empirical priors)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def generate_side_by_side_comparison(
    exhaustive_summary_path: Optional[str] = None,
    monte_carlo_summary_path: Optional[str] = None,
    output_path: Optional[str] = None,
    gated_media_path: Optional[str] = None,
    scans_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    from sim.config import ScenarioConfig, gated_packs
    from sim.exhaustive import gated_grid_profitability, scan_rates

    ex_path = exhaustive_summary_path or os.path.join(OUT, "exhaustive_summary.csv")
    mc_path = monte_carlo_summary_path or os.path.join(OUT, "monte_carlo_summary.csv")
    gm_path = gated_media_path or os.path.join(OUT, "gated_by_media.csv")
    out_path = output_path or os.path.join(OUT, "exhaustive_vs_realistic_side_by_side.csv")

    if not os.path.exists(ex_path):
        raise FileNotFoundError(f"Exhaustive summary not found at: {ex_path}")
    if not os.path.exists(mc_path):
        raise FileNotFoundError(f"Monte Carlo summary not found at: {mc_path}")

    ex_df = pd.read_csv(ex_path)
    mc_df = pd.read_csv(mc_path)
    gm_df = pd.read_csv(gm_path) if os.path.exists(gm_path) else None
    scans = scans_cache if scans_cache is not None else {}
    gated_by_key = {(p.platform, p.name): p for p in gated_packs() if p.price_bdt > 0}
    sc = ScenarioConfig()

    # Clean match keys
    rows = []
    # For every row in Monte Carlo summary (covers BL, Robi, and GP)
    for _, mc_row in mc_df.iterrows():
        platform = mc_row["platform"]
        pack_name = mc_row["pack"]
        c_size = float(mc_row.get("users_per_cohort", 100))

        # Match exhaustive row if token pack
        ex_match = ex_df[(ex_df["platform"] == platform) & (ex_df["pack"] == pack_name)]
        if len(ex_match) > 0:
            ex_row = ex_match.iloc[0]
            ex_combinations = int(ex_row.get("combinations", 0))
            ex_net_rev = float(ex_row.get("mean_net_bdt", np.nan))
            ex_mean_cogs = float(ex_row.get("mean_cogs_bdt", np.nan))
            ex_mean_contrib = float(ex_row.get("mean_contribution_bdt", np.nan))
            ex_p05_contrib = float(ex_row.get("p05_contribution_bdt", np.nan))
            ex_p50_contrib = float(ex_row.get("p50_contribution_bdt", np.nan))
            ex_p95_contrib = float(ex_row.get("p95_contribution_bdt", np.nan))
            ex_min_contrib = float(ex_row.get("min_contribution_bdt", np.nan))
            ex_max_contrib = float(ex_row.get("max_contribution_bdt", np.nan))
            ex_margin_mean = float(ex_row.get("gross_margin_mean", np.nan))
            ex_margin_p50 = float(ex_row.get("gross_margin_p50", np.nan))
            ex_prof_share = float(ex_row.get("profitable_share", np.nan))
            ex_breakage = float(ex_row.get("mean_breakage_memo_bdt", np.nan))
        else:
            # Gated packs (Banglalink & GP): evaluate full (media x file x 0.5M-5.5M chat x text_case) lattice
            ex_combinations = np.nan
            ex_prof_share = np.nan
            if gm_df is not None and (platform, pack_name) in gated_by_key:
                pack_obj = gated_by_key[(platform, pack_name)]
                sub_gm = gm_df[(gm_df["platform"] == platform) & (gm_df["pack"] == pack_name)]
                if len(sub_gm) > 0:
                    key = tuple(pack_obj.models)
                    if key not in scans:
                        scans[key] = scan_rates(pack_obj.models, scenario=sc)
                    ex_prof_share, ex_combinations = gated_grid_profitability(sub_gm, scans[key], scenario=sc)
            ex_net_rev = float(mc_row["revenue_bdt_mean"]) / c_size  # single user net
            ex_mean_cogs = np.nan
            ex_mean_contrib = np.nan
            ex_p05_contrib = np.nan
            ex_p50_contrib = np.nan
            ex_p95_contrib = np.nan
            ex_min_contrib = np.nan
            ex_max_contrib = np.nan
            ex_margin_mean = np.nan
            ex_margin_p50 = np.nan
            ex_breakage = 0.0

        # Monte Carlo values (per user basis = cohort / cohort_size)
        mc_net_rev = float(mc_row["revenue_bdt_mean"]) / c_size
        mc_mean_cogs = float(mc_row["cogs_bdt_mean"]) / c_size
        mc_p05_cogs = float(mc_row["cogs_bdt_p05"]) / c_size
        mc_p50_cogs = float(mc_row["cogs_bdt_p50"]) / c_size
        mc_p95_cogs = float(mc_row["cogs_bdt_p95"]) / c_size
        mc_mean_contrib = float(mc_row["contribution_bdt_mean"]) / c_size
        mc_p05_contrib = float(mc_row["contribution_bdt_p05"]) / c_size
        mc_p50_contrib = float(mc_row["contribution_bdt_p50"]) / c_size
        mc_p95_contrib = float(mc_row["contribution_bdt_p95"]) / c_size
        mc_min_contrib = float(mc_row.get("contribution_bdt_min", mc_row.get("worst_cohort_loss_bdt", np.nan))) / c_size
        mc_max_contrib = float(mc_row.get("contribution_bdt_max", np.nan)) / c_size
        mc_margin_mean = float(mc_row["gross_margin_mean"])
        mc_margin_p50 = float(mc_row["gross_margin_p50"])
        mc_margin_p05 = float(mc_row["gross_margin_p05"])
        mc_margin_p95 = float(mc_row["gross_margin_p95"])
        mc_prof_share = float(mc_row["profitable_cohort_share"])
        mc_breakage = float(mc_row["breakage_memo_bdt_mean"]) / c_size
        mc_gating_avoided = float(mc_row.get("gating_avoided_usd_per_1m_mean", np.nan))
        mc_gating_cheaper = float(mc_row.get("gating_share_whitelist_cheaper", np.nan))

        rows.append(
            {
                "platform": platform,
                "pack": pack_name,
                "ex_combinations": ex_combinations,
                "ex_net_rev_bdt": ex_net_rev,
                "mc_net_rev_bdt": mc_net_rev,
                "ex_mean_cogs_bdt": ex_mean_cogs,
                "mc_mean_cogs_bdt": mc_mean_cogs,
                "ex_mean_contrib_bdt": ex_mean_contrib,
                "mc_mean_contrib_bdt": mc_mean_contrib,
                "ex_p50_contrib_bdt": ex_p50_contrib,
                "mc_p50_contrib_bdt": mc_p50_contrib,
                "ex_p05_contrib_bdt": ex_p05_contrib,
                "mc_p05_contrib_bdt": mc_p05_contrib,
                "ex_p95_contrib_bdt": ex_p95_contrib,
                "mc_p95_contrib_bdt": mc_p95_contrib,
                "ex_min_contrib_bdt": ex_min_contrib,
                "mc_min_contrib_bdt": mc_min_contrib,
                "ex_max_contrib_bdt": ex_max_contrib,
                "mc_max_contrib_bdt": mc_max_contrib,
                "ex_gross_margin_mean": ex_margin_mean,
                "mc_gross_margin_mean": mc_margin_mean,
                "ex_gross_margin_p50": ex_margin_p50,
                "mc_gross_margin_p50": mc_margin_p50,
                "mc_gross_margin_p05": mc_margin_p05,
                "mc_gross_margin_p95": mc_margin_p95,
                "ex_profitable_share": ex_prof_share,
                "mc_profitable_cohort_share": mc_prof_share,
                "ex_mean_breakage_bdt": ex_breakage,
                "mc_mean_breakage_bdt": mc_breakage,
                "mc_gating_avoided_usd_per_1m": mc_gating_avoided,
                "mc_gating_share_whitelist_cheaper": mc_gating_cheaper,
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_path, index=False)
    return comp_df


if __name__ == "__main__":
    df = generate_side_by_side_comparison()
    print("Side-by-side comparison table generated successfully.")
    print(df.to_string())

