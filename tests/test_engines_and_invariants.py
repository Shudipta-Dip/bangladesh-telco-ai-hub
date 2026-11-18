"""Independent brute-force and invariant tests for simplex, exhaustive, Monte Carlo, and sensitivity engines."""

from __future__ import annotations

import unittest

import numpy as np

from sim.config import (
    IO_RATIOS,
    WEIGHT_PARTS,
    Pack,
    ScenarioConfig,
    active_catalog_models,
    banglalink_packs,
    build_models,
    gp_packs,
    robi_packs,
)
from sim.engine import MixBook
from sim.exhaustive import (
    evaluate_gp_pack,
    evaluate_token_pack,
    gated_grid_profitability,
    gp_media_cells,
    scan_gating,
    scan_rates,
    summarize_pack,
)
from sim.monte_carlo import CohortSampler, evaluate_cohorts
from sim.sensitivity import (
    generate_sensitivity_cache_manual,
    generate_sensitivity_io_dearest,
    generate_vulnerability_corners,
)
from sim.simplex import mix_count, mix_matrix


class TestEnginesAndInvariants(unittest.TestCase):
    """Tests engine logic independently via brute-force enumeration, differential checks, and invariants."""

    def test_simplex_lattice_exact_enumeration(self):
        for n_models in (1, 2, 3, 4):
            mat = mix_matrix(n_models, WEIGHT_PARTS)
            self.assertEqual(len(mat), mix_count(n_models, WEIGHT_PARTS))
            self.assertEqual(mat.shape[1], n_models)
            self.assertTrue(np.all(mat.sum(axis=1) == WEIGHT_PARTS))
            self.assertTrue(np.all(mat >= 0))
            self.assertEqual(len(np.unique(mat, axis=0)), len(mat))

    def test_exhaustive_scan_and_summarize_match_brute_force(self):
        models = ["gpt-6", "deepseek-v4.1", "gemini-3.8"]
        test_pack = Pack(
            platform="test",
            pack_id="tp100",
            name="Test Pool 100",
            price_bdt=100.0,
            tokens=2_000_000.0,
            models=models,
            kind="token_pool",
        )
        sc = ScenarioConfig()
        md = build_models(sc)
        scan = scan_rates(models, scenario=sc)

        brute_rates = []
        levels = [round(i * 0.1, 1) for i in range(11)]
        for i in range(WEIGHT_PARTS + 1):
            for j in range(WEIGHT_PARTS + 1 - i):
                k = WEIGHT_PARTS - i - j
                w = (i / WEIGHT_PARTS, j / WEIGHT_PARTS, k / WEIGHT_PARTS)
                for io in IO_RATIOS:
                    in_s = io / (io + 1.0)
                    out_s = 1.0 / (io + 1.0)
                    for ch in levels:
                        r = 0.0
                        for m_idx, mid in enumerate(models):
                            if w[m_idx] == 0:
                                continue
                            m = md[mid]
                            in_p = (1.0 - ch) * m.miss + ch * m.read_price
                            r += w[m_idx] * (in_s * in_p + out_s * m.output)
                        brute_rates.append(r / 1_000_000.0)

        self.assertEqual(scan["count"], len(brute_rates))
        self.assertAlmostEqual(scan["mean_rate"], sum(brute_rates) / len(brute_rates), places=14)
        self.assertAlmostEqual(scan["min_rate"], min(brute_rates), places=14)
        self.assertAlmostEqual(scan["max_rate"], max(brute_rates), places=14)
        self.assertEqual(int(scan["best"][0].max()), WEIGHT_PARTS)
        self.assertEqual(int(scan["worst"][0].max()), WEIGHT_PARTS)

        rows = evaluate_token_pack(test_pack, scan, tokens_per_image=40_000.0, scenario=sc)
        self.assertEqual(len(rows), 111)
        summ = summarize_pack(rows, scan=scan, scenario=sc)
        # "test" platform defaults to robi_vat_mode="inclusive" (Net = 100/1.15 * 0.935)
        net_expected = (100.0 / 1.15) * 0.935
        max_text_cogs = (2_000_000.0 * max(brute_rates) + 2_000_000.0 * (sc.infra_usd_per_1m / 1_000_000.0)) * sc.fx_buy
        max_img_cogs = (2_000_000.0 / 40_000.0) * sc.bl_image_usd * sc.fx_buy
        text_only_full = [r for r in rows if abs(r["utilization"] - 1.0) < 1e-9 and abs(r["image_fraction"]) < 1e-9][0]
        self.assertAlmostEqual(text_only_full["contribution_min"], net_expected - max_text_cogs, places=4)
        self.assertAlmostEqual(summ["min_contribution_bdt"], net_expected - max(max_text_cogs, max_img_cogs), places=4)
        self.assertAlmostEqual(summ["max_contribution_bdt"], net_expected, places=6)

        single_book = MixBook(["gpt-6"], sc)
        keys_sb, w_sb = single_book.dearest_blend(0.5)
        self.assertEqual(keys_sb, ["gpt-6"])
        self.assertAlmostEqual(float(w_sb.sum()), 1.0)

        gating_rows = scan_gating([test_pack], open_scan_models=["gpt-6", "gemini-3.8"], scenario=sc)
        self.assertEqual(len(gating_rows), 1)
        self.assertGreaterEqual(gating_rows[0]["mean_avoided_usd_per_1m"], -1e-9)

    def test_3d_media_grid_and_multivolume_gated_profitability(self):
        # Creator & Legend must expand to independent 11x11x11 = 1,331 media cells (images x video x audio)
        bl_creator = next(p for p in banglalink_packs() if p.pack_id == "creator")
        cells = gp_media_cells(bl_creator)
        self.assertEqual(len(cells), 11 * 11 * 11)

        # Brute-force check over multi-volume chat grid & image grid on mini GP pack
        models = ["gemini-3.1", "gpt-6"]
        mini_gp = Pack(
            platform="gp",
            pack_id="mini",
            name="Mini GP",
            price_bdt=50.0,
            tokens=None,
            images=5.0,
            models=models,
            kind="gated",
            image_usd=0.035,
        )
        sc = ScenarioConfig()
        md = build_models(sc)
        scan = scan_rates(models, scenario=sc)
        gated_rows = evaluate_gp_pack(mini_gp, scan, scenario=sc)
        vols = (1_000_000.0, 3_000_000.0)
        share, total_combos = gated_grid_profitability(gated_rows, scan, chat_volumes=vols, scenario=sc)

        levels = [round(i * 0.1, 1) for i in range(11)]
        net = 50.0 * 0.935  # GP added_on_top
        profitable = 0
        total = 0
        for img_f in levels:
            media_bdt = img_f * 5.0 * 0.035 * sc.fx_buy
            for c_tok in vols:
                for w0 in range(WEIGHT_PARTS + 1):
                    w = (w0 / WEIGHT_PARTS, (WEIGHT_PARTS - w0) / WEIGHT_PARTS)
                    for io in IO_RATIOS:
                        in_s, out_s = io / (io + 1.0), 1.0 / (io + 1.0)
                        for ch in levels:
                            rate = sum(
                                w[idx] * (in_s * ((1.0 - ch) * md[mid].miss + ch * md[mid].read_price) + out_s * md[mid].output)
                                for idx, mid in enumerate(models)
                            ) / 1_000_000.0
                            cogs = (c_tok * rate + c_tok * (sc.infra_usd_per_1m / 1e6)) * sc.fx_buy + media_bdt
                            if net >= cogs:
                                profitable += 1
                            total += 1

        self.assertEqual(total_combos, total)
        self.assertAlmostEqual(share, profitable / total, delta=0.001)

    def test_monte_carlo_differential_user_level_and_new_random_axes(self):
        sc = ScenarioConfig(volume_discount=0.10, infra_usd_per_1m=0.05)
        all_models = active_catalog_models()
        sampler = CohortSampler(n_cohorts=3, cohort_size=5, seed=99)
        behaviors = sampler.sample_behaviors(all_models, ScenarioConfig())
        md = build_models(sc)

        for pack in robi_packs() + banglalink_packs() + gp_packs():
            self.assertTrue(set(pack.models).issubset(set(all_models)), pack.key)

        robi_999 = next(p for p in robi_packs() if p.pack_id == "p999")
        bl_creator = next(p for p in banglalink_packs() if p.pack_id == "creator")
        gp_pro = next(p for p in gp_packs() if p.pack_id == "pro")

        # 1. Verify Robi modes + infra shift + inclusive VAT per user
        c_robi, s_robi = evaluate_cohorts(robi_999, behaviors, all_models, scenario=sc, n_cohorts=3, cohort_size=5)
        self.assertEqual(len(c_robi), 3)
        self.assertTrue(np.isnan(s_robi["gating_avoided_usd_per_1m_mean"]))
        mode_weights = {
            "saver": {"deepseek-v4-flash": 0.5, "gemini-2.5-flash-lite": 0.5},
            "balanced": {
                "deepseek-v4-flash": 0.175,
                "gemini-2.5-flash-lite": 0.175,
                "gpt-5-nano": 0.175,
                "grok-4.1-fast-nr": 0.175,
                "gemini-3.5-flash-lite": 0.10,
                "kimi-k2.6": 0.10,
                "claude-haiku-4.5": 0.05,
                "grok-4.3": 0.05,
            },
            "expert": {
                "deepseek-v4-flash": 0.05,
                "gemini-2.5-flash-lite": 0.05,
                "gpt-5-nano": 0.05,
                "grok-4.1-fast-nr": 0.05,
                "gemini-3.5-flash-lite": 0.15,
                "kimi-k2.6": 0.15,
                "claude-haiku-4.5": 0.25,
                "grok-4.3": 0.25,
            },
            "beast": {"claude-haiku-4.5": 0.5, "grok-4.3": 0.5},
        }
        mean_fx = float(np.mean(behaviors["fx_buy"]))
        tpi = (sc.robi_image_usd * mean_fx) / (999.0 / 20_000_000.0)
        manual_cogs_robi = []
        for u in range(15):
            wmap = mode_weights[str(behaviors["robi_mode"][u])]
            io = float(behaviors["io_ratio"][u])
            hit = float(behaviors["cache_hit"][u])
            wr = float(behaviors["cache_write"][u])
            miss_s = max(0.0, 1.0 - hit - wr)
            in_s, out_s = io / (io + 1.0), 1.0 / (io + 1.0)
            usd_tok = sum(
                w * (in_s * (miss_s * md[mid].miss + hit * md[mid].read_price + wr * md[mid].write_price) + out_s * md[mid].output)
                for mid, w in wmap.items()
            ) / 1_000_000.0
            budget = 20_000_000.0 * float(behaviors["utilization"][u])
            imgs = float(behaviors["image_fraction"][u]) * (budget / tpi)
            txt = max(0.0, budget - imgs * tpi)
            fx = float(behaviors["fx_buy"][u])
            tt = float(behaviors["tt_commission"][u])
            inf = float(behaviors["infra_per_1m"][u]) + (sc.infra_usd_per_1m - 0.02)
            text_bdt = (txt * usd_tok * 0.90 + txt * (inf / 1e6)) * fx * (1.0 + tt)
            img_bdt = imgs * sc.robi_image_usd * 0.90 * fx * (1.0 + tt)
            manual_cogs_robi.append(text_bdt + img_bdt)
        expected_c_cogs = np.array(manual_cogs_robi).reshape(3, 5).sum(axis=1)
        self.assertTrue(np.allclose(c_robi["cogs_bdt"].to_numpy(), expected_c_cogs, rtol=1e-9))

        # 1b. Verify Robi Tk 99 commercial mode confinement (expert & beast folded to balanced)
        robi_99 = next(p for p in robi_packs() if p.pack_id == "p99")
        c_robi99, _ = evaluate_cohorts(robi_99, behaviors, all_models, scenario=sc, n_cohorts=3, cohort_size=5)
        tpi_99 = (sc.robi_image_usd * mean_fx) / (99.0 / 1_000_000.0)
        manual_cogs_99 = []
        for u in range(15):
            raw_m = str(behaviors["robi_mode"][u])
            confined_m = "balanced" if raw_m in ("expert", "beast") else raw_m
            wmap = mode_weights[confined_m]
            io = float(behaviors["io_ratio"][u])
            hit = float(behaviors["cache_hit"][u])
            wr = float(behaviors["cache_write"][u])
            miss_s = max(0.0, 1.0 - hit - wr)
            in_s, out_s = io / (io + 1.0), 1.0 / (io + 1.0)
            usd_tok = sum(
                w * (in_s * (miss_s * md[mid].miss + hit * md[mid].read_price + wr * md[mid].write_price) + out_s * md[mid].output)
                for mid, w in wmap.items()
            ) / 1_000_000.0
            budget = 1_000_000.0 * float(behaviors["utilization"][u])
            imgs = float(behaviors["image_fraction"][u]) * (budget / tpi_99)
            txt = max(0.0, budget - imgs * tpi_99)
            fx = float(behaviors["fx_buy"][u])
            tt = float(behaviors["tt_commission"][u])
            inf = float(behaviors["infra_per_1m"][u]) + (sc.infra_usd_per_1m - 0.02)
            text_bdt = (txt * usd_tok * 0.90 + txt * (inf / 1e6)) * fx * (1.0 + tt)
            img_bdt = imgs * sc.robi_image_usd * 0.90 * fx * (1.0 + tt)
            manual_cogs_99.append(text_bdt + img_bdt)
        expected_c_cogs_99 = np.array(manual_cogs_99).reshape(3, 5).sum(axis=1)
        self.assertTrue(np.allclose(c_robi99["cogs_bdt"].to_numpy(), expected_c_cogs_99, rtol=1e-9))

        # 2. Verify Banglalink Creator (independent audio, 4-8s clip draw, file context tokens, 27-model gating)
        #    and GP Pro (continuous gp_image_usd draw, 27-model gating)
        for pack in (bl_creator, gp_pro):
            c_df, summ = evaluate_cohorts(pack, behaviors, all_models, scenario=sc, n_cohorts=3, cohort_size=5)
            pos_map = {m: idx for idx, m in enumerate(all_models)}
            cheapest_m = min(pack.models, key=lambda mid: (md[mid].output, md[mid].miss))
            manual_cogs = []
            manual_avoided = []
            vid_p = pack.video_usd_per_sec if pack.video_usd_per_sec is not None else sc.video_usd_per_sec
            aud_p = pack.audio_usd if pack.audio_usd is not None else 0.012
            for u in range(15):
                w_allowed = {mid: float(behaviors["model_weights"][u, pos_map[mid]]) for mid in pack.models}
                w_allowed[cheapest_m] += max(0.0, 1.0 - sum(w_allowed.values()))
                io = float(behaviors["io_ratio"][u])
                hit = float(behaviors["cache_hit"][u])
                wr = float(behaviors["cache_write"][u])
                miss_s = max(0.0, 1.0 - hit - wr)
                in_s, out_s = io / (io + 1.0), 1.0 / (io + 1.0)
                in_usd_1m = sum(
                    w * (miss_s * md[mid].miss + hit * md[mid].read_price + wr * md[mid].write_price)
                    for mid, w in w_allowed.items()
                )
                usd_1m = in_s * in_usd_1m + out_s * sum(w * md[mid].output for mid, w in w_allowed.items())
                open_1m = sum(
                    float(behaviors["model_weights"][u, idx]) * (
                        in_s * (miss_s * md[mid].miss + hit * md[mid].read_price + wr * md[mid].write_price)
                        + out_s * md[mid].output
                    )
                    for idx, mid in enumerate(all_models)
                )
                manual_avoided.append(open_1m - usd_1m)

                fx = float(behaviors["fx_buy"][u])
                tt = float(behaviors["tt_commission"][u])
                inf = float(behaviors["infra_per_1m"][u]) + (sc.infra_usd_per_1m - 0.02)
                seg_u = int(behaviors["segments"][u])
                if seg_u == 2:
                    chat_tok = sc.gated_power_chat_tokens_min + float(behaviors["utilization"][u]) * (sc.gated_power_chat_tokens_max - sc.gated_power_chat_tokens_min)
                else:
                    chat_tok = sc.gated_chat_tokens_base * float(behaviors["utilization"][u])
                text_bdt = (chat_tok * (usd_1m / 1e6) * 0.90 + chat_tok * (inf / 1e6)) * fx * (1.0 + tt)

                img_p = float(behaviors["gp_image_usd"][u]) if pack.platform == "gp" else pack.image_usd
                vid_cap = pack.video_clips * float(behaviors["bl_clip_sec"][u]) if pack.video_clips else pack.video_sec
                media_usd = (
                    float(behaviors["image_fraction"][u]) * pack.images * img_p
                    + float(behaviors["video_fraction"][u]) * vid_cap * vid_p
                    + float(behaviors["audio_fraction"][u]) * pack.audio * aud_p
                )
                media_bdt = media_usd * 0.90 * fx * (1.0 + tt)
                if pack.platform == "banglalink" and pack.files > 0:
                    f_tok = float(behaviors["file_fraction"][u]) * pack.files * float(behaviors["bl_tokens_per_file"][u])
                    file_bdt = (f_tok * (in_usd_1m / 1e6) * 0.90 + f_tok * (inf / 1e6)) * fx * (1.0 + tt)
                else:
                    file_bdt = 0.0
                manual_cogs.append(text_bdt + media_bdt + file_bdt)
            expected_cogs = np.array(manual_cogs).reshape(3, 5).sum(axis=1)
            self.assertTrue(np.allclose(c_df["cogs_bdt"].to_numpy(), expected_cogs, rtol=1e-9), pack.key)
            self.assertAlmostEqual(summ["gating_avoided_usd_per_1m_mean"], float(np.mean(manual_avoided)), places=9)

        # 3. Verify domain bounds on new Continuous Sampler axes
        b = sampler.sample_behaviors(all_models, ScenarioConfig())
        self.assertTrue(np.all((b["gp_image_usd"] >= 0.035) & (b["gp_image_usd"] <= 0.050)))
        self.assertTrue(np.all((b["bl_clip_sec"] >= 4.0) & (b["bl_clip_sec"] <= 8.0)))
        self.assertTrue(np.all((b["audio_fraction"] >= 0.0) & (b["audio_fraction"] <= 1.0)))
        self.assertTrue(np.all((b["file_fraction"] >= 0.0) & (b["file_fraction"] <= 1.0)))
        self.assertTrue(np.all((b["bl_tokens_per_file"] >= 15_000.0) & (b["bl_tokens_per_file"] <= 25_000.0)))
        # Audio & file fractions are drawn independently from image_fraction (not 1:1 locked)
        self.assertGreater(np.max(np.abs(b["audio_fraction"] - b["image_fraction"])), 0.1)
        self.assertGreater(np.max(np.abs(b["file_fraction"] - b["image_fraction"])), 0.1)

    def test_sensitivity_corners_and_monotonicity(self):
        packs = [
            next(p for p in banglalink_packs() if p.pack_id == "basic"),
            next(p for p in banglalink_packs() if p.pack_id == "creator"),
            next(p for p in robi_packs() if p.pack_id == "p999"),
            next(p for p in gp_packs() if p.pack_id == "lite"),
            next(p for p in gp_packs() if p.pack_id == "pro"),
        ]
        corners = generate_vulnerability_corners(packs)
        by_key = {(r["platform"], r["pack"]): r for _, r in corners.iterrows()}

        self.assertAlmostEqual(float(by_key[("robi", "Tk 999")]["contribution_bdt"]), (999.0 / 1.15) * 0.935 - 4981.32, places=3)
        self.assertGreater(float(by_key[("banglalink", "Basic Tk 100")]["contribution_bdt"]), 0.0)
        self.assertEqual(by_key[("gp", "Lite Tk 99")]["dearest_model"], "Gemini 3 Flash")

        sc = ScenarioConfig(volume_discount=0.15, bank_tt_commission_pct=0.02)
        df_cm = generate_sensitivity_cache_manual(packs, scenario=sc)
        df_io = generate_sensitivity_io_dearest(packs, scenario=sc)

        for pack in packs:
            sub_cm = df_cm[(df_cm["platform"] == pack.platform) & (df_cm["pack"] == pack.name)]
            self.assertEqual(len(sub_cm), 121)
            m1 = sub_cm[np.isclose(sub_cm["manual_share"], 1.0)].sort_values("cache_hit_rate")["contribution_bdt"].to_numpy()
            self.assertTrue(np.all(np.diff(m1) >= -1e-9), pack.key)

            sub_io = df_io[(df_io["platform"] == pack.platform) & (df_io["pack"] == pack.name)]
            self.assertEqual(len(sub_io), 66)
            d3 = sub_io[np.isclose(sub_io["io_ratio"], 3.0)].sort_values("dearest_model_weight")["contribution_bdt"].to_numpy()
            self.assertTrue(np.all(np.diff(d3) <= 1e-9), pack.key)


if __name__ == "__main__":
    unittest.main()
