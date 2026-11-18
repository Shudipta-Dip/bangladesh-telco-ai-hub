"""Independent spec-to-code verification against bd_info.md Sections 1, 3, 4, and 5."""

from __future__ import annotations

import math
import unittest

from sim.config import (
    BL_TOKENS_PER_FILE,
    BL_VIDEO_SEC_PER_CLIP,
    BTRC,
    DEFAULT_IMAGE_USD,
    FX_BUY,
    FX_WAR,
    GP_VIDEO_CAP_SEC,
    INFRA_USD_PER_1M,
    MDR,
    SOF,
    ScenarioConfig,
    active_catalog_models,
    banglalink_packs,
    build_models,
    gp_packs,
    robi_packs,
)

# Ground-truth image, video, and audio rates verbatim from bd_info.md Section 4 (USD)
IMG_RATES = {
    "flux-schnell": 0.003,
    "flux-2-pro": 0.040,
    "gpt-image-2": 0.040,
    "nano-banana-2-lite": 0.035,
    "nano-banana-2": 0.050,
    "seedream-5-pro": 0.035,
    "grok-imagine": 0.040,
}

VID_RATES = {
    "seedance-1-pro-fast": 0.060,
    "wan-2.2-5b-fast": 0.040,
    "hailuo-02-fast": 0.080,
    "lightricks-ltx-2.5": 0.080,
    "p-video-draft": 0.050,
}

AUD_RATES = {
    "minimax-music-1.5": 0.012,
    "flux-music": 0.012,
    "meta-musicgen": 0.012,
    "elevenlabs-voice": 0.012,
    "elevenlabs-music": 0.015,
}


class TestSpecAndCatalog(unittest.TestCase):
    """Verifies every pack, model rate, tier, VAT default, and BD macro constant matches bd_info.md."""

    def test_bangladesh_macro_cost_stack_and_defaults(self):
        self.assertAlmostEqual(FX_BUY, 123.30, places=6)
        self.assertAlmostEqual(FX_WAR, 123.20, places=6)
        self.assertAlmostEqual(BTRC, 0.055, places=6)
        self.assertAlmostEqual(SOF, 0.010, places=6)
        self.assertAlmostEqual(MDR, 0.0, places=6)
        self.assertAlmostEqual(INFRA_USD_PER_1M, 0.02, places=6)
        self.assertAlmostEqual(DEFAULT_IMAGE_USD, 0.035, places=6)
        self.assertAlmostEqual(GP_VIDEO_CAP_SEC, 30.0, places=6)
        self.assertAlmostEqual(BL_VIDEO_SEC_PER_CLIP, 5.0, places=6)
        self.assertAlmostEqual(BL_TOKENS_PER_FILE, 20_000.0, places=6)

        sc = ScenarioConfig()
        self.assertEqual(sc.gp_vat_mode, "added_on_top")
        self.assertEqual(sc.bl_vat_mode, "inclusive")
        self.assertEqual(sc.robi_vat_mode, "inclusive")

    def test_banglalink_ryze_packs_and_media_arithmetic_means(self):
        packs = {p.pack_id: p for p in banglalink_packs()}
        self.assertEqual(set(packs.keys()), {"basic", "advanced", "pro", "creator", "legend"})

        expected_specs = {
            "basic": {
                "price": 100.0,
                "files": 5.0,
                "images": 5.0,
                "video_sec": 0.0,
                "video_clips": None,
                "audio": 0.0,
                "llms": {"gemini-3.1", "gpt-6", "deepseek-v4.1"},
                "img_models": ["flux-schnell"],
                "vid_models": [],
                "aud_models": [],
            },
            "advanced": {
                "price": 300.0,
                "files": 15.0,
                "images": 15.0,
                "video_sec": 0.0,
                "video_clips": None,
                "audio": 0.0,
                "llms": {"grok-4.3", "gemini-3.1", "gpt-6", "deepseek-v4.1"},
                "img_models": ["flux-schnell"],
                "vid_models": [],
                "aud_models": [],
            },
            "pro": {
                "price": 500.0,
                "files": 25.0,
                "images": 54.0,
                "video_sec": 65.0,  # 13 videos x 5 sec
                "video_clips": 13.0,
                "audio": 0.0,
                "llms": {"gpt-6", "gemini-3.8", "claude-haiku-5.5", "grok-4.3", "deepseek-v4.1"},
                "img_models": ["flux-schnell", "flux-2-pro", "nano-banana-2-lite"],
                "vid_models": ["seedance-1-pro-fast", "wan-2.2-5b-fast"],
                "aud_models": [],
            },
            "creator": {
                "price": 1000.0,
                "files": 30.0,
                "images": 272.0,
                "video_sec": 325.0,  # 65 videos x 5 sec
                "video_clips": 65.0,
                "audio": 389.0,
                "llms": {
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
                },
                "img_models": ["flux-schnell", "gpt-image-2", "nano-banana-2", "flux-2-pro", "seedream-5-pro"],
                "vid_models": ["seedance-1-pro-fast", "hailuo-02-fast", "wan-2.2-5b-fast", "p-video-draft"],
                "aud_models": ["minimax-music-1.5", "flux-music", "meta-musicgen", "elevenlabs-voice"],
            },
            "legend": {
                "price": 1500.0,
                "files": 80.0,
                "images": 272.0,
                "video_sec": 325.0,  # 65 videos x 5 sec
                "video_clips": 65.0,
                "audio": 389.0,
                "llms": {
                    "gpt-6.1-terra",
                    "claude-sonnet-5",
                    "grok-4.6",
                    "gemini-3.8-flash",
                    "deepseek-v4-pro",
                    "qwen3.8-max",
                    "perplexity-sonar-pro",
                    "mistral-large-2512",
                },
                "img_models": [
                    "flux-schnell",
                    "gpt-image-2",
                    "nano-banana-2",
                    "flux-2-pro",
                    "seedream-5-pro",
                    "grok-imagine",
                ],
                "vid_models": [
                    "seedance-1-pro-fast",
                    "hailuo-02-fast",
                    "wan-2.2-5b-fast",
                    "lightricks-ltx-2.5",
                    "p-video-draft",
                ],
                "aud_models": [
                    "minimax-music-1.5",
                    "flux-music",
                    "meta-musicgen",
                    "elevenlabs-voice",
                    "elevenlabs-music",
                ],
            },
        }

        for pid, exp in expected_specs.items():
            pack = packs[pid]
            self.assertEqual(pack.kind, "gated")
            self.assertIsNone(pack.tokens)
            self.assertEqual(pack.validity_days, 30)
            self.assertAlmostEqual(pack.price_bdt, exp["price"])
            self.assertAlmostEqual(pack.files, exp["files"])
            self.assertAlmostEqual(pack.images, exp["images"])
            self.assertAlmostEqual(pack.video_sec, exp["video_sec"])
            self.assertEqual(pack.video_clips, exp["video_clips"])
            self.assertAlmostEqual(pack.audio, exp["audio"])
            self.assertEqual(set(pack.models), exp["llms"])

            total_modalities = len(exp["llms"]) + len(exp["img_models"]) + len(exp["vid_models"]) + len(exp["aud_models"])
            expected_total = {"basic": 4, "advanced": 5, "pro": 10, "creator": 24, "legend": 24}[pid]
            self.assertEqual(total_modalities, expected_total)

            mean_img = math.fsum(IMG_RATES[m] for m in exp["img_models"]) / len(exp["img_models"])
            self.assertAlmostEqual(pack.image_usd, mean_img, places=5)

            if exp["vid_models"]:
                mean_vid = math.fsum(VID_RATES[m] for m in exp["vid_models"]) / len(exp["vid_models"])
                self.assertAlmostEqual(pack.video_usd_per_sec, mean_vid, places=5)
            else:
                self.assertIsNone(pack.video_usd_per_sec)

            if exp["aud_models"]:
                mean_aud = math.fsum(AUD_RATES[m] for m in exp["aud_models"]) / len(exp["aud_models"])
                self.assertAlmostEqual(pack.audio_usd, mean_aud, places=5)
            else:
                self.assertIsNone(pack.audio_usd)

    def test_robi_everyai_packs(self):
        packs = {p.pack_id: p for p in robi_packs()}
        self.assertEqual(set(packs.keys()), {"p99", "p149", "p349", "p599", "p999"})
        expected = {
            "p99": (99.0, 1_000_000.0),
            "p149": (149.0, 2_000_000.0),
            "p349": (349.0, 5_000_000.0),
            "p599": (599.0, 10_000_000.0),
            "p999": (999.0, 20_000_000.0),
        }
        robi_llms = {
            "claude-haiku-4.5",
            "deepseek-v4-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash-lite",
            "kimi-k2.6",
            "gpt-5-nano",
            "grok-4.1-fast-nr",
            "grok-4.3",
        }
        for pid, (price, tokens) in expected.items():
            pack = packs[pid]
            self.assertEqual(pack.kind, "token_pool")
            self.assertEqual(pack.category, "monthly")
            self.assertEqual(pack.validity_days, 30)
            self.assertAlmostEqual(pack.price_bdt, price)
            self.assertAlmostEqual(pack.tokens, tokens)
            self.assertEqual(set(pack.models), robi_llms)

    def test_gp_oneai_packs_grok_mapping_and_tiered_images(self):
        packs = {p.pack_id: p for p in gp_packs()}
        self.assertEqual(set(packs.keys()), {"free", "student", "lite", "plus", "pro"})
        expected = {
            "free": (0.0, 0.0, 0.0, 0.035, {"gpt-5-nano", "gemini-3-flash"}),
            "student": (71.0, 10.0, 0.0, 0.035, {"gpt-5-nano", "gemini-3-flash"}),
            "lite": (99.0, 30.0, 0.0, 0.035, {"deepseek-v4-flash", "gemini-3-flash", "grok-4.1-fast-nr", "gpt-5-nano"}),
            "plus": (
                299.0,
                50.0,
                0.0,
                0.045,
                {"deepseek-v4-flash", "gemini-3-flash", "grok-4.3", "gpt-5-nano", "gemini-3.8-flash", "gpt-5-mini", "gpt-5.4", "kimi-k2.6"},
            ),
            "pro": (
                799.0,
                100.0,
                30.0,
                0.050,
                {"claude-sonnet-4", "claude-sonnet-5", "claude-opus-5", "gpt-5.6-sol", "grok-4.6", "deepseek-v4-pro"},
            ),
        }
        for pid, (price, images, vid_sec, img_usd, llms) in expected.items():
            pack = packs[pid]
            self.assertEqual(pack.kind, "gated")
            self.assertIsNone(pack.tokens)
            self.assertAlmostEqual(pack.price_bdt, price)
            self.assertAlmostEqual(pack.images, images)
            self.assertAlmostEqual(pack.video_sec, vid_sec)
            self.assertAlmostEqual(pack.image_usd, img_usd)
            self.assertEqual(set(pack.models), llms)

    def test_master_price_book_and_scenario_switches(self):
        models = build_models(ScenarioConfig())
        active = set(active_catalog_models())
        self.assertEqual(len(active), 27)
        for pack in robi_packs() + banglalink_packs() + gp_packs():
            for mid in pack.models:
                self.assertIn(mid, models)
                self.assertIn(mid, active)

        expected_rows = {
            "gpt-5-nano": (0.05, 0.005, 0.05, 0.40, "economy"),
            "gpt-6": (0.15, 0.015, 0.15, 0.60, "economy"),
            "gpt-5-mini": (0.25, 0.025, 0.25, 2.00, "mid"),
            "gpt-5.4": (2.50, 0.250, 2.50, 15.00, "flagship"),
            "gpt-5.6-sol": (4.00, 0.400, 5.00, 20.00, "flagship"),
            "gpt-6.1-terra": (2.00, 0.200, 2.00, 12.00, "flagship"),
            "claude-haiku-4.5": (1.00, 0.100, 1.25, 5.00, "economy"),
            "claude-haiku-5.5": (1.00, 0.100, 1.25, 5.00, "economy"),
            "claude-sonnet-4": (3.00, 0.300, 3.75, 15.00, "flagship"),
            "claude-sonnet-5": (2.00, 0.200, 2.50, 10.00, "mid"),
            "claude-opus-5": (5.00, 0.500, 6.25, 25.00, "flagship"),
            "gemini-2.5-flash-lite": (0.10, 0.010, 0.10, 0.40, "economy"),
            "gemini-3.1": (0.10, 0.010, 0.10, 0.40, "economy"),
            "gemini-3.5-flash-lite": (0.30, 0.030, 0.30, 2.50, "mid"),
            "gemini-3-flash": (0.50, 0.050, 0.50, 3.00, "mid"),
            "gemini-3.8": (0.75, 0.075, 0.75, 3.75, "mid"),
            "gemini-3.8-flash": (0.75, 0.075, 0.75, 3.75, "mid"),
            "grok-4.1-fast-nr": (0.20, 0.050, 0.20, 0.50, "economy"),
            "grok-4.3": (1.25, 0.200, 1.25, 2.50, "mid"),
            "grok-4.6": (2.00, 0.500, 2.00, 6.00, "mid"),
            "deepseek-v4-flash": (0.15, 0.003, 0.15, 0.60, "economy"),
            "deepseek-v4.1": (0.15, 0.003, 0.15, 0.60, "economy"),
            "deepseek-v4-pro": (0.66, 0.022, 0.66, 1.98, "mid"),
            "kimi-k2.6": (0.95, 0.160, 0.95, 4.00, "mid"),
            "qwen3.8-max": (2.00, 2.000, 2.00, 6.00, "mid"),
            "perplexity-sonar-pro": (3.00, 0.500, 3.00, 15.00, "flagship"),
            "mistral-large-2512": (2.00, 0.500, 2.00, 6.00, "mid"),
        }
        for mid, (miss, read, write, out, tier) in expected_rows.items():
            m = models[mid]
            self.assertAlmostEqual(m.miss, miss, msg=mid)
            self.assertAlmostEqual(m.read_price, read, msg=mid)
            self.assertAlmostEqual(m.write_price, write, msg=mid)
            self.assertAlmostEqual(m.output, out, msg=mid)
            self.assertEqual(m.tier, tier, msg=mid)

        s_cliff = build_models(ScenarioConfig(gemini_38_price="list_2027"))
        self.assertAlmostEqual(s_cliff["gemini-3.8-flash"].output, 7.50)
        self.assertAlmostEqual(s_cliff["gemini-3.8"].output, 7.50)

        s_peak = build_models(ScenarioConfig(deepseek_window="peak"))
        self.assertAlmostEqual(s_peak["deepseek-v4-pro"].output, 3.96)
        self.assertAlmostEqual(s_peak["deepseek-v4.1"].output, 1.20)

        s_ttl = build_models(ScenarioConfig(claude_cache_ttl="1h"))
        self.assertAlmostEqual(s_ttl["claude-haiku-4.5"].write_price, 2.00)
        self.assertAlmostEqual(s_ttl["claude-opus-5"].write_price, 10.00)

        s_lc = build_models(ScenarioConfig(long_context=True))
        self.assertAlmostEqual(s_lc["gpt-5.6-sol"].output, 30.00)
        self.assertAlmostEqual(s_lc["grok-4.6"].output, 12.00)


if __name__ == "__main__":
    unittest.main()
