# Optimized SR wall-time comparison

This experiment directly compares standard SR, bagged SR (K=4), and MS-SR
(K=4) using only the work required by each algorithm. It was run from the TFIM
step-2000 checkpoint with L=100 and Ns=12,000 samples per candidate batch on
one NVIDIA H100 80GB HBM3 GPU, hosted by a dual-socket Intel Xeon Platinum
8481C system. All methods use the same Hermitian
eigendecomposition-based pseudoinverse solver.

## Protocol

- Three device/seed pairs, with all three methods run on the same physical GPU
  within each pair.
- Five device-synchronized timed repetitions after two warm-ups per method and
  pair.
- Fixed checkpoint parameters during timing; the common final parameter write
  and optional residual diagnostics are excluded.
- Method-independent sampler names. For every timed repetition, SR's samples
  equal the first candidate batch of both K=4 methods, and all four sample
  fingerprints agree between bagged SR and MS-SR.
- SR uses one batch and one solve at lambda=1e-4. Bagged SR uses four batches
  and four solves at lambda=1e-4 followed by a uniform average. MS-SR uses four
  batches and four solves with spectrum quantiles {0.9, 0.7, 0.4, 0.1} and
  exact leave-one-batch-out (LOO) stacking.

## Direct implementation and optimizations

The benchmark bypasses the general research callback and executes exactly
1/4/4 candidate solves for SR/bagged SR/MS-SR. For MS-SR:

1. The first unshifted NTK eigendecomposition supplies the spectrum-derived
   shifts and is reused immediately for the first shifted solve; there is no
   separate spectrum decomposition.
2. Exact LOO reuses the four candidate batches and their already-computed local
   energies.
3. The 12 off-batch candidate directions are evaluated in four batched JVP
   calls. Their median time is 0.477 s, and the simplex fit takes 0.006 s.
4. Prediction sufficient statistics are reduced on device before transferring
   the small Gram matrices and target projections used by the simplex fit.

## Results

We first take the median over the five repetitions within each device/seed
pair. Paired differences and ratios are computed repetition-wise before that
median. We then report the median and range across the three independent
pairs.

| method | steady time, median | three-pair range | relative to SR |
|---|---:|---:|---:|
| SR | 13.3836 s | 13.1536–13.4493 s | 1.0000× |
| bagged SR (K=4) | 53.5488 s | 52.7056–53.5987 s | 3.9991× |
| MS-SR (K=4) | 54.0858 s | 53.1442–54.1786 s | 4.0389× |

The paired MS-SR increment over bagged SR is 0.5503 s (1.027%), with a
three-pair range of 0.4373–0.5790 s (0.829–1.080%). Thus both K=4 methods have
the same dominant budget—four sampled batches and four SR solves—while exact
MS-SR adds approximately 1% measured wall time.

## Files and reproduction

- `data/timing_summary.csv` contains the aggregate table.
- `data/paired_cluster_summary.csv` contains the three independent pair-level
  summaries.
- `data/pair{1,2,3}_{sr,bagged_sr_k4,ms_sr_k4}.json` contains all timed rows,
  component times, sample fingerprints, solver metadata, and process-lifetime
  device-memory high-water marks.
- `../../src/sr_filtering_nqs/large_scale/core/optimized_sr_benchmark.py`
  implements the direct update constructors.
- `../../src/sr_filtering_nqs/large_scale/cli/benchmark_optimized_sr_cost.py`
  is the benchmark CLI.
- `../../tests/test_optimized_sr_benchmark.py` contains focused
  numerical tests.

The exact checkpoint is not bundled because of its size; it can be produced by
the included TFIM training code. Example invocation from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  .venv/bin/python -m sr_filtering_nqs.large_scale.cli.benchmark_optimized_sr_cost \
  --checkpoint /path/to/checkpoint_step002000.pkl \
  --method mssr --seed 3 --warmup 2 --repeats 5 --k 4 --no-apply-update \
  --output rebuttal_assets/sr_cost/data/mssr_cost_example.json
```

The recorded environment used Python 3.12.13, JAX 0.7.2, NetKet 3.21.0, and
SciPy 1.18.0. The memory values in the JSON files are process-lifetime/JIT
high-water marks, not isolated per-step peaks.
