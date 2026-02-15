"""Inspect DSEC dataset: load voxel grids and save visualizations."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsec_root", type=Path, required=True, help="Path to DSEC root (must contain train/)")
    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--delta_t_ms", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--save_dir", type=Path, default=Path("artifacts/inspect"))
    args = parser.parse_args()

    # Make vendored DSEC example importable (so `import dsec_dataset.*` works)
    third_party_root = Path(__file__).resolve().parent.parent / "third_party" / "dsec_example"
    sys.path.insert(0, str(third_party_root))

    from dsec_dataset.provider import DatasetProvider  # noqa: E402

    args.save_dir.mkdir(parents=True, exist_ok=True)

    provider = DatasetProvider(args.dsec_root, delta_t_ms=args.delta_t_ms, num_bins=args.num_bins)
    ds = provider.get_train_dataset()
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(f"Dataset length: {len(ds)}")

    for bidx, batch in enumerate(dl):
        if bidx >= args.num_batches:
            break

        # batch['representation']['left'] shape: [B, C, H, W]
        vox_left = batch["representation"]["left"].float()
        disp = batch["disparity_gt"].float()

        # basic stats
        nonzero = (vox_left != 0).float().mean().item()
        print(f"Batch {bidx}: vox_left shape={tuple(vox_left.shape)} disp shape={tuple(disp.shape)} nonzero_frac={nonzero:.6f}")

        # visualize first element in batch
        v = vox_left[0]
        ev_img = torch.sum(v, dim=0).cpu().numpy()  # [H,W]
        ev_img = ev_img / (np.max(np.abs(ev_img)) + 1e-6)

        d = disp[0].cpu().numpy()

        # save images
        plt.figure()
        plt.title(f"event-sum bins | batch {bidx}")
        plt.imshow(ev_img, cmap="gray")
        plt.axis("off")
        plt.savefig(args.save_dir / f"event_sum_b{bidx}.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.title(f"disparity | batch {bidx}")
        plt.imshow(d, cmap="inferno")
        plt.axis("off")
        plt.savefig(args.save_dir / f"disparity_b{bidx}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved visualizations to: {args.save_dir}")


if __name__ == "__main__":
    main()
