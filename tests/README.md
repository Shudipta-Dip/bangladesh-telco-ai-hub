# Independent Verification Suite (`tests/`)

This test suite checks every layer of the BD Telco AI Hub Unit Economics Simulator **independently from the `sim/` code**, deriving expected numbers directly from plain arithmetic and [`bd_info.md`](../bd_info.md).

## How to Run

Run from the repository root (`Simulation BD Telco`):

```powershell
python -u -m unittest discover -s tests -v
```

## What Each Test Module Proves

1. **[`test_catalog_and_spec.py`](test_catalog_and_spec.py)**
   - Verifies every Banglalink (`video_clips`, `files`, media means), Robi (`8` text LLMs), and Grameenphone (`grok-4.1-fast-nr` on Lite, `grok-4.3` on Plus, tiered `$0.035/$0.045/$0.050` image rates) pack definition against [`bd_info.md`](../bd_info.md).
   - Verifies all `27` active model rates/tiers, per-operator VAT defaults (`gp_vat_mode="added_on_top"`, `bl_vat_mode="inclusive"`, `robi_vat_mode="inclusive"`), `bl_video_sec_per_clip=5.0`, `bl_tokens_per_file=20,000`, and upstream price switches.

2. **[`test_independent_accounting.py`](test_independent_accounting.py)**
   - Implements revenue, per-operator VAT (including `50/50` channel split), token-pool debit, file input-context tokens, COGS, and contribution from scratch without calling `sim` formula helpers for expected values.
   - Asserts exact agreement on hand-calculated users (Robi `Tk 999` `-4,169.09 BDT`, `-18.7%` inclusive-VAT + BTRC spread invariance, Banglalink `Basic` & `Creator` with files & `8s/clip` switches, GP `Lite` `Gemini 3 Flash` dearest model, and stressed GP `Pro`).

3. **[`test_engines_and_invariants.py`](test_engines_and_invariants.py)**
   - Compares `scan_rates()`, 3D `(image × video × audio)` `gp_media_cells()`, and multi-volume `gated_grid_profitability()` against pure-Python brute-force enumeration.
   - Re-evaluates simulated Monte Carlo users element-by-element (verifying `Uniform(0.035, 0.050)` GP images, `Uniform(4, 8)` BL video clips, independent audio/file draws, file input context tokens, and `27`-model active gating avoidance).
   - Verifies sensitivity monotonicity and worst-case vulnerability corners.

4. **[`test_outputs_and_plots_reconciliation.py`](test_outputs_and_plots_reconciliation.py)**
   - Cross-checks all CSVs in [`../output/`](../output) for completeness, exact accounting identities, CRN stress monotonicity (`0` spurious noise on unaffected packs), and `27`-model MC gating columns.
   - Captures rendered Matplotlib strings directly from [`sim/plots.py`](../sim/plots.py) in a temp directory to verify all `7` charts (including `-Tk 18.7` Robi formatting, `4.93×` Creator, `27`-model gating avoidance, and clean text formatting) without touching production PNGs, plus checks `>= 20px` RGBA margins on [`../output/figures/`](../output/figures).
