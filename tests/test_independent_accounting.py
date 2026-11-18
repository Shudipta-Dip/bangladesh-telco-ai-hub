"""Independent from-scratch mathematical oracle for bd_info.md Section 6 equations."""

from __future__ import annotations

import unittest

import numpy as np

from sim.config import (
    ROBI_IMAGE_USD,
    ScenarioConfig,
    banglalink_packs,
    gp_packs,
    robi_packs,
    tokens_per_image_for_pack,
)
from sim.engine import (
    contribution_at_volume,
    default_tokens_per_image,
    effective_video_sec,
    gp_result,
    net_revenue_bdt,
    recognized_and_cash,
    token_pool_result,
)
from sim.exhaustive import project_weights
from sim.pricing import one_hot, tier_mix


class TestIndependentAccountingOracle(unittest.TestCase):
    """Compares sim.pricing and sim.engine against standalone textbook arithmetic."""

    def test_per_operator_vat_and_net_revenue_modes(self):
        gp_pro = next(p for p in gp_packs() if p.pack_id == "pro")
        bl_cr = next(p for p in banglalink_packs() if p.pack_id == "creator")
        robi_999 = next(p for p in robi_packs() if p.pack_id == "p999")

        # Baseline per-operator defaults: GP adds +15% on top; BL & Robi are VAT-inclusive
        rec_gp, cash_gp = recognized_and_cash(gp_pro)
        self.assertAlmostEqual(rec_gp, 799.0)
        self.assertAlmostEqual(cash_gp, 799.0 * 1.15)

        for p in (bl_cr, robi_999):
            r, c = recognized_and_cash(p)
            self.assertAlmostEqual(r, p.price_bdt / 1.15)
            self.assertAlmostEqual(c, p.price_bdt)

        # Switching BL/Robi to talktime recharge must NOT mutate GP added_on_top
        sc_talk = ScenarioConfig(bl_vat_mode="sticker_is_revenue", robi_vat_mode="sticker_is_revenue")
        r_gp2, c_gp2 = recognized_and_cash(gp_pro, scenario=sc_talk)
        self.assertAlmostEqual(r_gp2, 799.0)
        self.assertAlmostEqual(c_gp2, 799.0 * 1.15)
        r_bl2, _ = recognized_and_cash(bl_cr, scenario=sc_talk)
        self.assertAlmostEqual(r_bl2, 1000.0)

        # 50/50 channel split blends inclusive & sticker_is_revenue cleanly while GP stays insulated
        sc_split = ScenarioConfig(bl_vat_mode="mixed_50_50", robi_vat_mode="mixed_50_50")
        r_sp, _ = recognized_and_cash(bl_cr, scenario=sc_split)
        self.assertAlmostEqual(r_sp, 0.5 * 1000.0 + 0.5 * (1000.0 / 1.15), places=8)
        self.assertAlmostEqual(recognized_and_cash(gp_pro, scenario=sc_split)[0], 799.0)

        # Net revenue keeps (1 - 0.055 - 0.01) * (1 - mdr) * (1 - churn)
        net_stressed = float(net_revenue_bdt(1000.0, btrc=0.055, sof=0.01, mdr=0.0185, churn_refund=0.03))
        expected_net = 1000.0 * (1.0 - 0.055 - 0.01) * (1.0 - 0.0185) * (1.0 - 0.03)
        self.assertAlmostEqual(net_stressed, expected_net, places=10)

    def test_robi_tk999_worst_case_text_hand_calculation(self):
        # Independent calculation under inclusive VAT default:
        # Recognized = 999 / 1.15, Net = (999 / 1.15) * 0.935 = 812.23043478 BDT
        # Worst text COGS (Claude Haiku 4.5, 3:1, 0% cache, 20M tokens) = 4981.32 BDT
        robi_999 = next(p for p in robi_packs() if p.pack_id == "p999")
        net_expected = (999.0 / 1.15) * 0.935
        cogs_expected = (20_000_000 / 1_000_000.0) * (0.75 * 1.00 + 0.25 * 5.00 + 0.02) * 123.30
        contrib_expected = net_expected - cogs_expected  # -4169.089565 BDT

        res = token_pool_result(
            pack=robi_999,
            scenario_name="all_dearest",
            utilization=1.0,
            cache_hit=0.0,
            manual_share=1.0,
            io_ratio=3.0,
            image_fraction=0.0,
        )
        self.assertAlmostEqual(float(res["net_bdt"]), net_expected, places=6)
        self.assertAlmostEqual(float(res["cogs_bdt"]), cogs_expected, places=6)
        self.assertAlmostEqual(float(res["contribution_bdt"]), contrib_expected, places=6)

    def test_robi_tk999_100pct_image_vat_btrc_spread_invariant(self):
        # Mathematical identity under inclusive VAT + token-pool image debit:
        # Wholesale Image COGS = PriceBDT, NetRev = (PriceBDT / 1.15) * 0.935
        # -> Contribution per Tk 100 sticker = 100 * (0.935 / 1.15 - 1.0) = -18.695652... BDT (identically across ALL packs)
        robi_999 = next(p for p in robi_packs() if p.pack_id == "p999")
        expected_tpi = (ROBI_IMAGE_USD * 123.30) / (999.0 / 20_000_000.0)
        self.assertAlmostEqual(tokens_per_image_for_pack(robi_999), expected_tpi, places=6)

        res = token_pool_result(
            pack=robi_999,
            scenario_name="all_dearest",
            utilization=1.0,
            cache_hit=0.0,
            manual_share=1.0,
            io_ratio=3.0,
            image_fraction=1.0,
        )
        self.assertAlmostEqual(float(res["text_tokens"]), 0.0, places=6)
        self.assertAlmostEqual(float(res["cogs_bdt"]), 999.0, places=6)
        self.assertAlmostEqual(100.0 * float(res["contribution_bdt"]) / 999.0, 100.0 * (0.935 / 1.15 - 1.0), places=6)

    def test_null_token_fallback_guard(self):
        bl_basic = next(p for p in banglalink_packs() if p.pack_id == "basic")
        gp_lite = next(p for p in gp_packs() if p.pack_id == "lite")
        sc = ScenarioConfig()
        # Gated packs return operator-specific multimodal frontier image tokens without cross-operator fallback or ValueError
        self.assertAlmostEqual(tokens_per_image_for_pack(bl_basic, scenario=sc), sc.bl_tokens_per_image, places=6)
        self.assertAlmostEqual(tokens_per_image_for_pack(gp_lite, scenario=sc), sc.gp_tokens_per_image, places=6)
        expected_robi_default = (ROBI_IMAGE_USD * 123.30) / (99.0 / 1_000_000.0)
        self.assertAlmostEqual(default_tokens_per_image(None), expected_robi_default, places=6)

    def test_banglalink_basic_creator_files_and_clip_length(self):
        # Basic Tk 100 worst case (inclusive VAT, 2M tokens, 5 images, 5 files * 20k input)
        bl_basic = next(p for p in banglalink_packs() if p.pack_id == "basic")
        res_basic = gp_result(
            bl_basic,
            "all_dearest",
            cache_hit=0.0,
            manual_share=1.0,
            io_ratio=3.0,
            image_fraction=1.0,
            file_fraction=1.0,
        )
        net_basic = (100.0 / 1.15) * 0.935  # 81.304348 BDT
        media_basic = 5.0 * 0.003 * 123.30  # 1.8495 BDT
        file_basic = (100_000.0 / 1e6) * (0.15 + 0.02) * 123.30  # 2.0961 BDT (GPT 6 uncached miss + infra)
        text_basic = 2.0 * (0.2625 + 0.02) * 123.30  # 69.6645 BDT
        self.assertAlmostEqual(float(res_basic["file_bdt"]), file_basic, places=6)
        c_basic = float(
            contribution_at_volume(
                float(res_basic["gated_usd_per_1m"]) / 1e6,
                2_000_000.0,
                float(res_basic["net_bdt"]),
                float(res_basic["media_bdt"]) + float(res_basic["file_bdt"]),
            )
        )
        self.assertAlmostEqual(c_basic, net_basic - media_basic - file_basic - text_basic, places=4)

        # Creator Tk 1,000 clip length switch (5s -> 8s) and full media + 30 files
        bl_creator = next(p for p in banglalink_packs() if p.pack_id == "creator")
        self.assertAlmostEqual(effective_video_sec(bl_creator), 325.0)
        self.assertAlmostEqual(effective_video_sec(bl_creator, ScenarioConfig(bl_video_sec_per_clip=8.0)), 520.0)

        res_cr = gp_result(
            bl_creator,
            "all_dearest",
            cache_hit=0.0,
            manual_share=1.0,
            io_ratio=3.0,
            image_fraction=1.0,
            video_fraction=1.0,
            audio_fraction=1.0,
            file_fraction=1.0,
        )
        expected_media_bdt = (272 * 0.0336 + 325 * 0.0575 + 389 * 0.012) * 123.30
        expected_file_bdt = (30 * 20_000 / 1e6) * (3.00 + 0.02) * 123.30  # Sonar Pro miss $3 + infra $0.02
        expected_text_bdt = 2.0 * (6.00 + 0.02) * 123.30
        expected_corner = (1000.0 / 1.15) * 0.935 - expected_media_bdt - expected_file_bdt - expected_text_bdt
        self.assertAlmostEqual(float(res_cr["media_bdt"]), expected_media_bdt, places=4)
        self.assertAlmostEqual(float(res_cr["file_bdt"]), expected_file_bdt, places=4)
        self.assertAlmostEqual(float(res_cr["break_even_tokens"]), 0.0, places=6)
        c_cr = float(
            contribution_at_volume(
                float(res_cr["gated_usd_per_1m"]) / 1e6,
                2_000_000.0,
                float(res_cr["net_bdt"]),
                float(res_cr["media_bdt"]) + float(res_cr["file_bdt"]),
            )
        )
        self.assertAlmostEqual(c_cr, expected_corner, places=3)

    def test_gp_pro_stressed_and_gp_lite_grok41_dearest(self):
        gp_pro = next(p for p in gp_packs() if p.pack_id == "pro")
        sc = ScenarioConfig(
            volume_discount=0.20,
            infra_usd_per_1m=0.08,
            bank_tt_commission_pct=0.025,
            mdr=0.0185,
            churn_refund=0.03,
            claude_cache_ttl="1h",
        )
        res = gp_result(
            gp_pro,
            "all_dearest",
            cache_hit=0.0,
            manual_share=1.0,
            io_ratio=3.0,
            image_fraction=1.0,
            video_fraction=1.0,
            cache_write_share=0.2,
            scenario=sc,
        )
        self.assertEqual(res["dearest_id"], "claude-opus-5")
        in_share, out_share = 0.75, 0.25
        expected_usd_1m = in_share * (0.8 * 5.0 + 0.2 * 10.0) + out_share * 25.0
        net_exp = 799.0 * 0.935 * (1.0 - 0.0185) * (1.0 - 0.03)
        media_exp = (100 * 0.050 + 30 * 0.10) * 0.80 * 123.30 * 1.025  # GP Pro image = $0.050
        text_exp = (2.0 * expected_usd_1m * 0.80 + 2.0 * 0.08) * 123.30 * 1.025
        contrib = float(
            contribution_at_volume(
                float(res["gated_usd_per_1m"]) / 1e6,
                2_000_000.0,
                float(res["net_bdt"]),
                float(res["media_bdt"]),
                scenario=sc,
            )
        )
        self.assertAlmostEqual(contrib, net_exp - media_exp - text_exp, places=5)

    def test_one_hot_tier_mix_and_projection_rules(self):
        # With grok-4.1-fast-nr ($0.20/$0.50) mapped onto GP Lite, Gemini 3 Flash ($0.50/$3.00 -> $1.125) is now dearest
        gp_lite = next(p for p in gp_packs() if p.pack_id == "lite")
        self.assertEqual(one_hot(gp_lite.models, "dearest"), {"gemini-3-flash": 1.0})
        self.assertEqual(one_hot(gp_lite.models, "cheapest"), {"gpt-5-nano": 1.0})

        # GP Plus keeps grok-4.3 alongside GPT-5.4 (flagship -> dearest)
        gp_plus = next(p for p in gp_packs() if p.pack_id == "plus")
        self.assertIn("grok-4.3", gp_plus.models)
        self.assertEqual(one_hot(gp_plus.models, "dearest"), {"gpt-5.4": 1.0})

        # On GP Lite (3 economy + 1 mid, 0 flagship): missing 10% flagship folds onto economy (0.80 / 3)
        mixed = tier_mix(gp_lite.models, {"economy": 0.70, "mid": 0.20, "flagship": 0.10})
        self.assertAlmostEqual(sum(mixed.values()), 1.0, places=10)
        self.assertAlmostEqual(mixed["gpt-5-nano"], 0.80 / 3.0, places=10)
        self.assertAlmostEqual(mixed["deepseek-v4-flash"], 0.80 / 3.0, places=10)
        self.assertAlmostEqual(mixed["grok-4.1-fast-nr"], 0.80 / 3.0, places=10)
        self.assertAlmostEqual(mixed["gemini-3-flash"], 0.20, places=10)

        # Open weights projection onto basic pack sums cleanly to 1.0
        bl_basic = next(p for p in banglalink_packs() if p.pack_id == "basic")
        open_ids = ["gpt-6.1-terra", "gpt-6", "claude-opus-5", "gemini-3.1"]
        w = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])
        proj = project_weights(open_ids, bl_basic.models, w)
        self.assertTrue(np.allclose(proj.sum(axis=1), 1.0))
        self.assertAlmostEqual(proj[0, 0], 0.75)
        self.assertAlmostEqual(proj[1, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
