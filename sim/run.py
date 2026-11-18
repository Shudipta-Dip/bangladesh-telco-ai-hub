"""Master runner for the BD Telco AI Hub Unit Economics Simulation Suite.

Executes:
1. Exhaustive factor grid (deterministic boundary search across billions of combinations)
2. Continuous Monte Carlo cohort simulation (10,000 cohorts x 100 users from empirical priors)
3. Sensitivity analysis tables (cache vs manual share, I:O ratio vs dearest model weight)
4. Utilization-band P&L ([0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0])
5. Vulnerability corners (explicit stress-test points for GP Pro, Robi Tk 999, etc.)
6. Regulatory and upstream pricing stress tests (Gemini 2027 cliff, DeepSeek peak, discounts)
7. Visualizations (updating all figures in output/figures)
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time

import numpy as np
import pandas as pd

from sim.config import (
    FACTOR_STEP,
    FX_BUY,
    FX_WAR,
    IMAGE_USD,
    IO_RATIOS,
    MODELS,
    VIDEO_USD_PER_SEC,
    WEIGHT_PARTS,
    Pack,
    ScenarioConfig,
    banglalink_tokens_per_image,
    build_models,
    gp_packs,
    banglalink_packs,
    gated_packs,
    token_packs,
    tokens_per_image_for_pack,
)
from sim.exhaustive import (
    evaluate_gp_pack,
    evaluate_token_pack,
    factor_levels,
    mix_label,
    scan_gating,
    scan_rates,
    summarize_pack,
)
from sim.monte_carlo import run_monte_carlo_simulation
from sim.comparison import generate_side_by_side_comparison
from sim.sensitivity import (
    generate_grid_by_utilization_bands,
    generate_sensitivity_cache_manual,
    generate_sensitivity_io_dearest,
    generate_vulnerability_corners,
    run_scenario_stress_test,
)
from sim.simplex import mix_count, mix_matrix

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _checks():
    small = mix_matrix(2, WEIGHT_PARTS)
    if len(small) != mix_count(2, WEIGHT_PARTS) or len(small) != WEIGHT_PARTS + 1:
        raise RuntimeError("two-model lattice has the wrong number of rows")
    if len(np.unique(small, axis=0)) != len(small):
        raise RuntimeError("lattice contains a duplicate mix")
    if not np.all(small.sum(axis=1) == WEIGHT_PARTS):
        raise RuntimeError("lattice rows do not sum to one")
    levels = factor_levels()
    if len(levels) != 11 or abs(levels[1] - levels[0] - FACTOR_STEP) > 1e-12:
        raise RuntimeError("factor step is not tenths")
    if len(IO_RATIOS) != 6:
        raise RuntimeError("input:output set changed")


def _money(value):
    if value != value or value is None:
        return "n/a"
    return f"{value:,.0f}"


def _pct(value):
    if value != value or value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _tokens(value):
    if value != value or value is None:
        return "n/a"
    if value <= 0:
        return "media already exceeds net revenue"
    return f"{value:,.0f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    _checks()
    t_start = time.time()
    scenario = ScenarioConfig(name="Baseline")
    models_dict = build_models(scenario)

    print("=" * 80)
    print("BD TELCO AI HUB UNIT ECONOMICS: FULL ADVERSARIAL SIMULATION SUITE")
    print("=" * 80)
    print(
        f"FX Buy: {scenario.fx_buy:.2f} BDT  |  FX WAR: {scenario.fx_war:.2f} BDT  |  "
        f"VAT: GP={scenario.gp_vat_mode}, BL={scenario.bl_vat_mode}, Robi={scenario.robi_vat_mode}"
    )
    print(f"BTRC Revenue Share: {scenario.btrc * 100:.1f}%  |  Social Obligation Fund: {scenario.sof * 100:.1f}%  |  BL clip={scenario.bl_video_sec_per_clip:.0f}s, {scenario.bl_tokens_per_file:,.0f} tok/file")
    print(f"Catalog contains {len(models_dict)} verified models with published cache miss, read, and write rates.")
    print()

    # ---------------------------------------------------------
    # PART 1: DETERMINISTIC EXHAUSTIVE FACTOR GRID
    # ---------------------------------------------------------
    print("--- PART 1: DETERMINISTIC EXHAUSTIVE FACTOR GRID ---")
    print("Evaluating all simplex mix vectors x 6 I:O ratios x 11 cache levels x media/file/chat-volume grids...")
    t0 = time.time()

    usage_rows = []
    summary_rows = []
    extreme_rows = []
    scans = {}

    for pack in token_packs():
        key = tuple(pack.models)
        if key not in scans:
            print(f"  Generating lattice for {pack.platform} ({len(pack.models)} models, {mix_count(len(pack.models), WEIGHT_PARTS):,} mixes)...")
            scans[key] = scan_rates(pack.models, scenario=scenario)
        scan = scans[key]
        unit_img_price = scenario.robi_image_usd if pack.platform == "robi" else scenario.bl_image_usd
        tokens_per_img = tokens_per_image_for_pack(pack, fx_buy=scenario.fx_buy, image_usd=unit_img_price)
        rows = evaluate_token_pack(pack, scan, tokens_per_img, scenario=scenario)
        usage_rows.extend(rows)
        summary_rows.append(summarize_pack(rows, scan=scan, scenario=scenario))

        if key not in {row.get("catalog_key") for row in extreme_rows}:
            worst_mix, worst_io, worst_cache, worst_rate = scan["worst"]
            best_mix, best_io, best_cache, best_rate = scan["best"]
            extreme_rows.append(
                {
                    "catalog_key": key,
                    "platform": pack.platform,
                    "pack": "catalog",
                    "dearest_mix": mix_label(pack.models, worst_mix, models_dict),
                    "dearest_io_ratio": worst_io,
                    "dearest_cache_hit": worst_cache,
                    "dearest_usd_per_1m": worst_rate * 1_000_000.0,
                    "cheapest_mix": mix_label(pack.models, best_mix, models_dict),
                    "cheapest_io_ratio": best_io,
                    "cheapest_cache_hit": best_cache,
                    "cheapest_usd_per_1m": best_rate * 1_000_000.0,
                }
            )

    gated_rows = []
    gated_list = [p for p in gated_packs() if p.price_bdt > 0]
    for pack in gated_list:
        key = tuple(pack.models)
        print(f"  Generating lattice for {pack.platform.upper()} {pack.name} ({len(pack.models)} models, {mix_count(len(pack.models), WEIGHT_PARTS):,} mixes)...")
        if key not in scans:
            scans[key] = scan_rates(pack.models, scenario=scenario)
        scan = scans[key]
        gated_rows.extend(evaluate_gp_pack(pack, scan, scenario=scenario))
        worst_mix, worst_io, worst_cache, worst_rate = scan["worst"]
        best_mix, best_io, best_cache, best_rate = scan["best"]
        extreme_rows.append(
            {
                "platform": pack.platform,
                "pack": pack.name,
                "dearest_mix": mix_label(pack.models, worst_mix, models_dict),
                "dearest_io_ratio": worst_io,
                "dearest_cache_hit": worst_cache,
                "dearest_usd_per_1m": worst_rate * 1_000_000.0,
                "cheapest_mix": mix_label(pack.models, best_mix, models_dict),
                "cheapest_io_ratio": best_io,
                "cheapest_cache_hit": best_cache,
                "cheapest_usd_per_1m": best_rate * 1_000_000.0,
            }
        )

    print("  Comparing open-catalog demand vs gated whitelists (gating analysis)...")
    import gc
    gc.collect()
    gating_rows = scan_gating(gated_list, scenario=scenario)
    gc.collect()

    for row in extreme_rows:
        row.pop("catalog_key", None)

    usage = pd.DataFrame(usage_rows)
    summary = pd.DataFrame(summary_rows)
    gated_df = pd.DataFrame(gated_rows)
    gating = pd.DataFrame(gating_rows)
    extremes = pd.DataFrame(extreme_rows)

    usage.to_csv(os.path.join(OUT, "exhaustive_by_usage.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "exhaustive_summary.csv"), index=False)
    gated_df.to_csv(os.path.join(OUT, "gated_by_media.csv"), index=False)
    gated_df[gated_df["platform"] == "gp"].to_csv(os.path.join(OUT, "gp_by_media.csv"), index=False)
    gating.to_csv(os.path.join(OUT, "gating_breakdown.csv"), index=False)
    gating[gating["pack"].str.contains("Student|Lite|Plus|Pro Tk 799", regex=True)].to_csv(os.path.join(OUT, "gp_gating.csv"), index=False)
    extremes.to_csv(os.path.join(OUT, "extreme_mixes.csv"), index=False)

    print(f"  Exhaustive grid completed in {time.time() - t0:.1f}s.")
    print()

    # Print summary table
    p_summary = summary.copy()
    for col in ("mean_contribution_bdt", "min_contribution_bdt", "max_contribution_bdt", "mean_net_bdt", "mean_breakage_memo_bdt"):
        p_summary[col] = p_summary[col].map(_money)
    p_summary["profitable_share"] = p_summary["profitable_share"].map(_pct)
    p_summary["combinations"] = p_summary["combinations"].map(lambda v: f"{int(v):,}")
    print(p_summary[["platform", "pack", "combinations", "mean_net_bdt", "mean_contribution_bdt", "min_contribution_bdt", "max_contribution_bdt", "profitable_share"]].to_string(index=False))
    print()

    # ---------------------------------------------------------
    # PART 2: CONTINUOUS MONTE CARLO COHORT SIMULATION
    # ---------------------------------------------------------
    print("--- PART 2: CONTINUOUS EMPIRICAL MONTE CARLO (10,000 COHORTS x 100 USERS) ---")
    t1 = time.time()
    mc_summary, mc_cohorts = run_monte_carlo_simulation(n_cohorts=10_000, cohort_size=100, scenario=scenario, seed=42)
    mc_summary.to_csv(os.path.join(OUT, "monte_carlo_summary.csv"), index=False)
    mc_cohorts.to_csv(os.path.join(OUT, "monte_carlo_cohorts.csv"), index=False)
    del mc_cohorts
    gc.collect()
    print(f"  Monte Carlo simulation (1,000,000 users evaluated) completed in {time.time() - t1:.1f}s.")
    print()

    p_mc = mc_summary.copy()
    for col in ("revenue_bdt_mean", "cogs_bdt_mean", "contribution_bdt_mean", "contribution_bdt_p05", "contribution_bdt_p50", "contribution_bdt_p95", "worst_cohort_loss_bdt", "worst_single_user_loss_bdt", "breakage_memo_bdt_mean"):
        p_mc[col] = p_mc[col].map(_money)
    for col in ("gross_margin_mean", "gross_margin_p05", "gross_margin_p50", "gross_margin_p95", "profitable_cohort_share", "mean_unprofitable_user_share"):
        p_mc[col] = p_mc[col].map(_pct)

    print(p_mc[["platform", "pack", "revenue_bdt_mean", "cogs_bdt_mean", "contribution_bdt_mean", "gross_margin_mean", "gross_margin_p05", "gross_margin_p95", "profitable_cohort_share", "worst_single_user_loss_bdt"]].to_string(index=False))
    print()

    # ---------------------------------------------------------
    # PART 2.5: UNIFIED SIDE-BY-SIDE COMPARISON TABLE
    # ---------------------------------------------------------
    print("--- PART 2.5: UNIFIED SIDE-BY-SIDE COMPARISON (EXHAUSTIVE GRID vs REALISTIC MONTE CARLO) ---")
    comp_df = generate_side_by_side_comparison(scans_cache=scans)
    print("  Comparison saved to output/exhaustive_vs_realistic_side_by_side.csv.")
    print()

    # ---------------------------------------------------------
    # PART 3: SENSITIVITY MATRICES & UTILIZATION BANDS
    # ---------------------------------------------------------
    print("--- PART 3: SENSITIVITY TABLES & UTILIZATION BANDS ---")
    t2 = time.time()
    all_packs = token_packs() + [p for p in gated_packs() if p.price_bdt > 0]

    sens_cache = generate_sensitivity_cache_manual(all_packs, scenario)
    sens_cache.to_csv(os.path.join(OUT, "sensitivity_cache_manual.csv"), index=False)

    sens_io = generate_sensitivity_io_dearest(all_packs, scenario)
    sens_io.to_csv(os.path.join(OUT, "sensitivity_io_dearest.csv"), index=False)

    bands = generate_grid_by_utilization_bands(token_packs(), scenario)
    bands.to_csv(os.path.join(OUT, "grid_by_band.csv"), index=False)

    corners = generate_vulnerability_corners(all_packs, scenario)
    corners.to_csv(os.path.join(OUT, "vulnerability_corners.csv"), index=False)

    stress = run_scenario_stress_test(all_packs)
    stress.to_csv(os.path.join(OUT, "scenario_stress_test.csv"), index=False)

    print(f"  Sensitivity matrices and stress tests generated in {time.time() - t2:.1f}s.")
    print()

    print("VULNERABILITY CORNERS (Worst-case user consumption: 100% dearest model, 0% cache hit, 3:1 I:O, full media):")
    p_corners = corners.copy()
    for col in ("price_bdt", "net_bdt", "total_cogs_bdt", "contribution_bdt"):
        p_corners[col] = p_corners[col].map(_money)
    p_corners["gross_margin"] = p_corners["gross_margin"].map(_pct)
    p_corners["dearest_rate_usd_per_1m"] = p_corners["dearest_rate_usd_per_1m"].map(lambda v: f"${v:.2f}")
    print(p_corners[["platform", "pack", "price_bdt", "dearest_model", "dearest_rate_usd_per_1m", "total_cogs_bdt", "contribution_bdt", "gross_margin"]].to_string(index=False))
    print()

    # ---------------------------------------------------------
    # PART 4: CHARTS GENERATION
    # ---------------------------------------------------------
    print("--- PART 4: GENERATING PLAIN-LANGUAGE EXECUTIVE CHARTS ---")
    try:
        from sim.plots import main as generate_plots
        generate_plots()
        print("  All charts updated successfully in output/figures.")
    except Exception as e:
        print(f"  Notice during plot generation: {e}")

    print()
    print("=" * 80)
    print(f"SIMULATION SUITE COMPLETED IN {time.time() - t_start:.1f}s.")
    print(f"All datasets and reports stored in: {OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
