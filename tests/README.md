# Test Suite

This test suite verifies calculations, plan catalogs, simulator logic, and output files across the project.

## Running Tests

Run all unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

## Test Files

1. **`test_catalog_and_spec.py`**: Validates plan definitions, model pricing tiers, Bangladesh tax and regulatory rates, and scenario toggles.
2. **`test_independent_accounting.py`**: Computes revenue, wholesale API bills, and margins using independent arithmetic, then compares them against simulator outputs.
3. **`test_engines_and_invariants.py`**: Checks factor grid combinations, Monte Carlo sampling ranges, and worst-case user scenarios against brute-force loops.
4. **`test_outputs_and_plots_reconciliation.py`**: Confirms that all generated CSV files and chart figures match each other and follow safe visual margins.
