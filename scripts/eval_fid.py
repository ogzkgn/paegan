"""Compute FID for a saved generator checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import compute_fid_for_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FID for a PAE-GAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--real-root", type=Path, default=Path("celeba"), help="Path to the real image directory.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fid"), help="Directory for cached reals, generated fakes, and FID results.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of real and fake samples to compare.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation and FID.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for generator inference and FID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_fid_for_checkpoint(
        checkpoint_path=args.checkpoint,
        real_root=args.real_root,
        output_root=args.output_root,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
