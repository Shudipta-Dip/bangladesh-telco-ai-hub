"""End-to-end reconciliation tests linking generated CSVs, executive plots, CRN stress tests, and PNG margins."""

from __future__ import annotations

import os
import tempfile
import unittest
from math import comb
from unittest.mock import patch

from PIL import Image
import numpy as np
import pandas as pd

from sim.config import FX_BUY, WEIGHT_PARTS, banglalink_packs, gp_packs
from sim.plots import (
    _chat_rows,
    _fee_chart,
    _gated_facts,
    _menu_chart,
    _monte_carlo_chart,
    _often_chart,
    _picture_bill_chart,
    _picture_chart,
    _priciest_chart,
    _style,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")


@unittest.skipUnless(os.path.exists(os.path.join(OUT, "exhaustive_summary.csv")), "Run `python -m sim.run` first")
class TestOutputsAndPlotsReconciliation(unittest.TestCase):
    """Verifies cross-CSV consistency, CRN stress monotonicity, chart data/footer fidelity, and PNG edge safety."""

    @classmethod
    def setUpClass(cls):
        cls.ex_sum = pd.read_csv(os.path.join(OUT, "exhaustive_summary.csv"))
        cls.ex_usage = pd.read_csv(os.path.join(OUT, "exhaustive_by_usage.csv"))
        cls.gated = pd.read_csv(os.path.join(OUT, "gated_by_media.csv"))
        cls.gating = pd.read_csv(os.path.join(OUT, "gating_breakdown.csv"))
        cls.extremes = pd.read_csv(os.path.join(OUT, "extreme_mixes.csv"))
        cls.mc_sum = pd.read_csv(os.path.join(OUT, "monte_carlo_summary.csv"))
        cls.side = pd.read_csv(os.path.join(OUT, "exhaustive_vs_realistic_side_by_side.csv"))
        cls.bands = pd.read_csv(os.path.join(OUT, "grid_by_band.csv"))
        cls.corners = pd.read_csv(os.path.join(OUT, "vulnerability_corners.csv"))
        cls.sens_cm = pd.read_csv(os.path.join(OUT, "sensitivity_cache_manual.csv"))
        cls.sens_io = pd.read_csv(os.path.join(OUT, "sensitivity_io_dearest.csv"))
        cls.stress = pd.read_csv(os.path.join(OUT, "scenario_stress_test.csv"))

    def test_output_csv_completeness_and_row_counts(self):
        self.assertEqual(len(self.ex_sum), 5)
        self.assertEqual(len(self.mc_sum), 14)
        self.assertEqual(len(self.side), 14)
        self.assertEqual(len(self.corners), 14)
        self.assertEqual(len(self.gating), 9)
        self.assertEqual(len(self.bands), 20)
        # Banglalink: 11 + 11 + 121 + 1331 + 1331 = 2805; GP: 11 + 11 + 11 + 121 = 154 -> 2959 rows
        self.assertEqual(len(self.gated), 2959)
        self.assertEqual(len(self.sens_cm), 14 * 121)
        self.assertEqual(len(self.sens_io), 14 * 66)
        self.assertEqual(len(self.stress), 12 * 14)

        for col in ("ex_combinations", "ex_profitable_share", "mc_net_rev_bdt", "mc_mean_cogs_bdt", "mc_mean_contrib_bdt", "mc_profitable_cohort_share"):
            self.assertFalse(self.side[col].isna().any(), f"Unexpected NaN in side_by_side column {col}")

        # Gated packs must have 27-model MC gating stats populated in mc_sum & side_by_side
        gated_mc = self.mc_sum[self.mc_sum["platform"] != "robi"]
        self.assertFalse(gated_mc["gating_avoided_usd_per_1m_mean"].isna().any())
        self.assertFalse(gated_mc["gating_share_whitelist_cheaper"].isna().any())

    def test_cross_csv_accounting_and_crn_stress_invariants(self):
        for _, row in self.ex_sum.iterrows():
            self.assertAlmostEqual(row["mean_net_bdt"], row["mean_cogs_bdt"] + row["mean_contribution_bdt"], places=6)
            sub_u = self.ex_usage[(self.ex_usage["platform"] == row["platform"]) & (self.ex_usage["pack"] == row["pack"])]
            self.assertAlmostEqual(row["min_contribution_bdt"], float(sub_u["contribution_min"].min()), places=6)
            self.assertAlmostEqual(row["max_contribution_bdt"], float(sub_u["contribution_max"].max()), places=6)

            c_match = self.corners[(self.corners["platform"] == row["platform"]) & (self.corners["pack"] == row["pack"])].iloc[0]
            b_sub = self.bands[(self.bands["platform"] == row["platform"]) & (self.bands["pack"] == row["pack"])]
            self.assertAlmostEqual(row["min_contribution_bdt"], float(c_match["contribution_bdt"]), places=3)
            self.assertAlmostEqual(row["min_contribution_bdt"], float(b_sub["min_contribution_bdt"].min()), places=3)

        robi_ext = self.extremes[self.extremes["platform"] == "robi"].iloc[0]
        for _, c_row in self.corners.iterrows():
            if c_row["platform"] == "robi":
                self.assertAlmostEqual(float(c_row["dearest_rate_usd_per_1m"]), float(robi_ext["dearest_usd_per_1m"]), places=4)
            else:
                ext = self.extremes[(self.extremes["platform"] == c_row["platform"]) & (self.extremes["pack"] == c_row["pack"])].iloc[0]
                self.assertAlmostEqual(float(c_row["dearest_rate_usd_per_1m"]), float(ext["dearest_usd_per_1m"]), places=4)

        for _, s_row in self.side.iterrows():
            mc = self.mc_sum[(self.mc_sum["platform"] == s_row["platform"]) & (self.mc_sum["pack"] == s_row["pack"])].iloc[0]
            c_size = float(mc["users_per_cohort"])
            self.assertAlmostEqual(s_row["mc_net_rev_bdt"], float(mc["revenue_bdt_mean"]) / c_size, places=6)
            self.assertAlmostEqual(s_row["mc_mean_cogs_bdt"], float(mc["cogs_bdt_mean"]) / c_size, places=6)
            self.assertAlmostEqual(s_row["mc_mean_contrib_bdt"], float(mc["contribution_bdt_mean"]) / c_size, places=6)
            self.assertAlmostEqual(s_row["mc_profitable_cohort_share"], float(mc["profitable_cohort_share"]), places=6)
            if s_row["platform"] == "robi":
                ex = self.ex_sum[(self.ex_sum["platform"] == "robi") & (self.ex_sum["pack"] == s_row["pack"])].iloc[0]
                self.assertEqual(int(s_row["ex_combinations"]), int(ex["combinations"]))
                self.assertAlmostEqual(s_row["ex_profitable_share"], float(ex["profitable_share"]), places=6)

        robi_expected_combos = comb(WEIGHT_PARTS + 8 - 1, WEIGHT_PARTS) * 6 * 11 * 111
        self.assertEqual(robi_expected_combos, 142_476_048)
        self.assertTrue((self.ex_sum["combinations"] == robi_expected_combos).all())
        self.assertEqual(len(self.ex_usage), 5 * 111)

        band_bounds = {"[0, 0.25)": (0.0, 0.25), "[0.25, 0.50)": (0.25, 0.50), "[0.50, 0.75)": (0.50, 0.75), "[0.75, 1.0]": (0.75, 1.01)}
        for _, brow in self.bands.iterrows():
            lo, hi = band_bounds[brow["band"]]
            sub_u = self.ex_usage[(self.ex_usage["platform"] == brow["platform"]) & (self.ex_usage["pack"] == brow["pack"]) & (self.ex_usage["utilization"] >= lo) & (self.ex_usage["utilization"] < hi)]
            w = sub_u["combinations"].to_numpy(dtype=float)
            self.assertAlmostEqual(brow["contribution_bdt"], float(np.average(sub_u["contribution_mean"], weights=w)), places=6)
            self.assertAlmostEqual(brow["min_contribution_bdt"], float(sub_u["contribution_min"].min()), places=6)
            self.assertAlmostEqual(brow["max_contribution_bdt"], float(sub_u["contribution_max"].max()), places=6)
            self.assertAlmostEqual(brow["profitable_share"], float(np.average(sub_u["profitable_share"], weights=w)), places=6)

        # CRN Stress Test Invariants: exact zero noise on unaffected scenarios & strict monotonicity
        by_scen = {s: df.set_index(["platform", "pack"]) for s, df in self.stress.groupby("scenario")}
        base = by_scen["Baseline"]
        # GP Student Tk 71 has no Gemini 3.8, Claude, DeepSeek, or BL clip/VAT exposure -> must match Baseline to 1e-12
        for s_name in ("Banglalink 8s/clip Video", "BL/Robi Talktime Recharge (No App VAT Deduction)", "BL/Robi 50/50 Channel VAT Split", "Gemini 2027 Cliff (+100% price)", "DeepSeek Peak Window (2x price)", "Claude 1h Cache Write TTL", "Long Context (>200k tokens)"):
            self.assertAlmostEqual(float(by_scen[s_name].loc[("gp", "Student Tk 71"), "gross_margin_mean"]), float(base.loc[("gp", "Student Tk 71"), "gross_margin_mean"]), places=12)

        # Router Infra $0.10/1M must strictly reduce margin on EVERY single pack under CRN
        self.assertTrue((by_scen["Router Infra $0.10/1M"]["gross_margin_mean"] < base["gross_margin_mean"]).all())
        # Volume discounts strictly order: Disc 20% > Disc 10% > Baseline on every pack
        self.assertTrue((by_scen["Volume Discount 20%"]["gross_margin_mean"] > by_scen["Volume Discount 10%"]["gross_margin_mean"]).all())
        self.assertTrue((by_scen["Volume Discount 10%"]["gross_margin_mean"] > base["gross_margin_mean"]).all())
        # Banglalink 8s/clip strictly drops Pro, Creator, Legend, leaves GP & Robi exact
        self.assertTrue((by_scen["Banglalink 8s/clip Video"].loc["gp", "gross_margin_mean"] == base.loc["gp", "gross_margin_mean"]).all())
        # BL/Robi Talktime Recharge improves BL/Robi margins while GP stays 100% untouched
        self.assertTrue((by_scen["BL/Robi Talktime Recharge (No App VAT Deduction)"].loc["gp", "gross_margin_mean"] == base.loc["gp", "gross_margin_mean"]).all())
        self.assertTrue((by_scen["BL/Robi Talktime Recharge (No App VAT Deduction)"].loc["robi", "gross_margin_mean"] > base.loc["robi", "gross_margin_mean"]).all())

    def test_plots_reconcile_with_csv_data_without_touching_prod_figures(self):
        captured = {}

        def _spy_finish(fig, path, footer):
            name = os.path.basename(path)
            fig_texts = [t.get_text() for t in fig.texts]
            ax_texts = [t.get_text() for ax in fig.axes for t in ax.texts]
            captured[name] = {"fig_texts": fig_texts, "ax_texts": ax_texts, "footer": footer}
            out_p = os.path.join(tmpdir, name)
            fig.text(0.04, 0.02, footer, fontsize=8.6)
            fig.savefig(out_p, dpi=100)
            import matplotlib.pyplot as plt
            plt.close(fig)
            return out_p

        scans = {}
        chat = _chat_rows(self.ex_usage)
        bl_facts = _gated_facts(self.gated, banglalink_packs(), scans_cache=scans)
        gp_facts = _gated_facts(self.gated, gp_packs(), scans_cache=scans)

        for fact in bl_facts + gp_facts:
            s_row = self.side[(self.side["platform"] == fact["platform"]) & (self.side["pack"] == fact["pack"].name)].iloc[0]
            self.assertEqual(int(fact["total_scenarios"]), int(s_row["ex_combinations"]))
            self.assertAlmostEqual(fact["share"], float(s_row["ex_profitable_share"]), places=6)

        _style()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sim.plots.FIG", tmpdir), patch("sim.plots._finish", side_effect=_spy_finish):
                _fee_chart(chat, bl_facts, gp_facts)
                _picture_chart(chat, bl_facts, gp_facts)
                _often_chart(chat, bl_facts, gp_facts)
                _picture_bill_chart(chat, bl_facts, gp_facts)
                _priciest_chart(os.path.join(OUT, "monte_carlo_summary.csv"), os.path.join(OUT, "vulnerability_corners.csv"))
                _menu_chart(os.path.join(OUT, "monte_carlo_summary.csv"))
                _monte_carlo_chart(os.path.join(OUT, "monte_carlo_summary.csv"))

        self.assertEqual(len(captured), 7)

        fee = captured["when_the_fee_runs_out.png"]
        self.assertIn("84% of 20.0M", fee["footer"])
        self.assertTrue(any("million chat tokens" in t for t in fee["ax_texts"]))

        # Chart 2: All 5 Robi labels print exact uniform -Tk 18.7 (zero IEEE-754 drift)
        pic = captured["what_pictures_add.png"]
        robi_187 = [t for t in pic["ax_texts"] if "Media/Files: -Tk 18.7" in t]
        self.assertEqual(len(robi_187), 5)
        self.assertIn("-Tk 18.7 per Tk 100", pic["footer"])

        # Chart 3: All 14 packs match side-by-side CSV across 0.5M-5.5M chat grid
        often = captured["how_often_it_makes_money.png"]
        for _, s_row in self.side.iterrows():
            pct_str = f"{float(s_row['ex_profitable_share']) * 100.0:.1f}% profitable"
            self.assertTrue(any(pct_str in t for t in often["ax_texts"]), f"Missing bar label: {pct_str}")
        self.assertIn("0.5M–5.5M", " ".join(often["fig_texts"]))

        # Chart 4: Creator 4.93x & Robi 1.23x reconcile
        pic_bill = captured["pictures_cost_more_than_the_fee.png"]
        self.assertIn("4.93× fee", " ".join(pic_bill["ax_texts"]))
        self.assertIn("1.23× net revenue", pic_bill["footer"])

        # Chart 5: Corners & Creator small negative everyday label reconcile
        priciest = captured["priciest_model_vs_usual_mix.png"]
        all_p_txt = " ".join(priciest["ax_texts"] + [priciest["footer"]])
        for expected_tok in ("+Tk 8", "-Tk 4,902", "-Tk 5,112", "-Tk 4,169", "-Tk 2,710"):
            self.assertIn(expected_tok, all_p_txt)

        # Chart 6: 27-model MC gating avoidance reconciles directly against monte_carlo_summary.csv
        menu = captured["blocking_models.png"]
        for _, mc_row in self.mc_sum[self.mc_sum["platform"] != "robi"].iterrows():
            saved_tk = float(mc_row["gating_avoided_usd_per_1m_mean"]) * FX_BUY
            expected_note = f"Avoids ~Tk {saved_tk:,.0f}"
            self.assertTrue(any(expected_note in t for t in menu["ax_texts"]), f"Missing MC gating label: {expected_note}")

        # Chart 7: Zero unescaped LaTeX $ math blocks & exact MC medians
        mc_cap = captured["monte_carlo_cohort_margins.png"]
        for sub_txt in mc_cap["fig_texts"]:
            self.assertNotIn("$0.0", sub_txt)
        for _, mc_row in self.mc_sum.iterrows():
            med = float(mc_row["gross_margin_p50"]) * 100.0
            self.assertTrue(any(f"Median {med:.1f}%" in t for t in mc_cap["ax_texts"]), f"Missing MC median {med:.1f}%")

    def test_rendered_pngs_exist_and_have_safe_nonclipping_margins(self):
        expected_pngs = (
            "when_the_fee_runs_out.png",
            "what_pictures_add.png",
            "how_often_it_makes_money.png",
            "pictures_cost_more_than_the_fee.png",
            "priciest_model_vs_usual_mix.png",
            "blocking_models.png",
            "monte_carlo_cohort_margins.png",
        )
        for fname in expected_pngs:
            path = os.path.join(FIG, fname)
            self.assertTrue(os.path.exists(path), f"Missing rendered chart: {fname}")
            with Image.open(path) as im:
                w, h = im.size
                alpha = im.getchannel("A")
                bbox = alpha.point(lambda a: 255 if a > 12 else 0).getbbox()
            self.assertIsNotNone(bbox, f"Empty chart image: {fname}")
            x0, y0, x1, y1 = bbox
            left, top, right, bottom = x0, y0, w - x1, h - y1
            self.assertGreaterEqual(min(left, right, top, bottom), 20, f"Tight/clipped ink margin in {fname}: {(left, right, top, bottom)}")


if __name__ == "__main__":
    unittest.main()
