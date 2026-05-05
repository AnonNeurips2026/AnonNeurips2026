# SR Filtering NQS Anonymous Review Package

This repository is a history-free, anonymous review package containing the code and curated artifacts needed for the paper experiments. It intentionally excludes the manuscript source, full checkpoint sweeps, raw delta banks, external archive mirrors, virtual environments, caches, and local logs.

The bundle follows the rendered figure set in the manuscript. Placeholder-only material and exploratory candidate summaries are omitted.

## Contents

- `src/sr_filtering_nqs/`: small exact experiments, matched-state experiments, large-scale diagnostics, and plotting utilities.
- `src/nqs_support_core/`: minimal support utilities extracted for this package.
- `src/advanced_drivers/`, `src/netket_foundational/`, `src/nqs_nets/`, `src/spin_vmc/`: driver, model, network, and spin-system code used by the experiments.
- `artifacts/paper_figures/`: final rendered paper figures with PDF metadata scrubbed.
- `artifacts/small_scale/`: selected exact-result pickles for the four-panel small-scale figure.
- `artifacts/large_scale/`: compact CSV summaries for the large-scale claims.

## Quickstart

```bash
python3 scripts/audit_anonymity.py
python -m venv .venv
. .venv/bin/activate
pip install -e .
python -m compileall src scripts
```

The large-scale training scripts target accelerator environments and are included for reproducibility. The bundled large-scale artifacts are compact summaries rather than raw checkpoints or delta banks.

## Reproduce The Curated Figure-Style Panel

```bash
mkdir -p artifacts/reproduced
python -m sr_filtering_nqs.small_scale.plot_four_panel_draft \
  --matched-noise artifacts/small_scale/matched_noise/rbm490_vit10.pkl \
  --matched-state artifacts/small_scale/matched_state/alternating_polish_vit_step0100_to_direct_rbm_a2_figure1_style.pkl \
  --output artifacts/reproduced/figure2_four_panel.pdf
```

Large-scale summary plots can be regenerated from the bundled CSVs with:

```bash
python -m sr_filtering_nqs.large_scale.cli.plot_curated_artifacts \
  --figure all \
  --output-dir artifacts/reproduced
```
