from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.fixed_data_protocol import FIXED_DATA_REPEAT_COUNT, run_fixed_data_fig3  # noqa: E402


SYSTEM_CHOICES = ("all", "j1j2", "tfim", "ssm", "shastry")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 20260422 fixed-data Figure 3 workflow")
    parser.add_argument("--input-dir", default=str(ROOT / "large_scale"))
    parser.add_argument("--system", choices=SYSTEM_CHOICES, default="all")
    parser.add_argument("--source-train-lambda", type=float, default=1e-4)
    parser.add_argument("--source-steps", type=str, default="all")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=FIXED_DATA_REPEAT_COUNT)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--validation-round-samples", type=int, default=None)
    parser.add_argument("--validation-sample-chunk", type=int, default=None)
    parser.add_argument("--jvp-chunk", type=int, default=None)
    parser.add_argument("--delta-batch-size", type=int, default=10)
    parser.add_argument("--train-seed-base", type=int, default=None)
    parser.add_argument("--validation-seed-base", type=int, default=None)
    parser.add_argument("--spectrum-seed-base", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = ("j1j2", "tfim") if args.system == "all" else (args.system,)

    for system_name in systems:
        print(
            f"[Figure 3] system={system_name} repeat_count={args.repeat_count} k={args.k}",
            flush=True,
        )
        run_fixed_data_fig3(
            args.input_dir,
            system_name=system_name,
            source_training_lambda=float(args.source_train_lambda),
            source_steps=args.source_steps,
            latest_only=args.latest_only,
            repeat_count=int(args.repeat_count),
            k=int(args.k),
            n_val=args.n_val,
            validation_round_samples=args.validation_round_samples,
            validation_sample_chunk=args.validation_sample_chunk,
            jvp_chunk=args.jvp_chunk,
            delta_batch_size=int(args.delta_batch_size),
            train_seed_base=args.train_seed_base,
            validation_seed_base=args.validation_seed_base,
            spectrum_seed_base=args.spectrum_seed_base,
        )


if __name__ == "__main__":
    main()
