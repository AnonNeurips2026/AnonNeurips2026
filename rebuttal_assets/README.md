# Rebuttal assets (author response, July 2026)

## br2500/
Three-way online comparison on 8x8 J1-J2 (J2=0.5, no-symmetry ViT). One SR
trajectory is branched at iteration 2500; SR, SPRING, and MS-SR (K=4) each
receive the same extra budget of 500 units (one MS-SR update = 4 units).

- `energy_vs_budget_br2500.{png,pdf}` — panel (a): shared trajectory and the
  three continuations vs cumulative budget (one MS-SR step = 4 units); panel (b): the same branches vs extra
  budget, with high-precision final-state energy evaluations at budget 500.
- `data/shared_trajectory_1951_2500.csv`, `data/{sr,spring,mssr}_branch_*.csv`
  — per-step training-estimate energies per site (8192 samples per step).
- `data/final_state_eval_*.json` — independent final-state evaluations
  (fresh chains, reseeded and rethermalized each round; 4,194,304 samples per
  run, two independent runs each for MS-SR and SR, 8,388,608 total per method).

## checkpoint_local/
Summary data for the new checkpoint-local tables in the response.

- `bagged_lambda_summary_tfim.csv` — bagged SR at fixed shifts on the 10 TFIM
  checkpoints, batch-paired to the shipped ablation.
- `k_quantile_ablation_summary.csv` — K in {2,3,5} and quantile-grid ablations.
- `lt_trees_summary.csv` — protocol re-run on checkpoints trained at
  lambda_train = 1e-3 and 1e-5.
