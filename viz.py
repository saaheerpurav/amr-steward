"""
viz.py — AMR-Steward reward curve visualisation.

Reads checkpoints/amr-grpo/stage*/log_history.json produced by train.py
and saves reward curve plots to the same checkpoint directory.

Usage:
    python viz.py                          # uses default checkpoint dir
    python viz.py --output-dir path/to/checkpoints/amr-grpo
    python viz.py --show                   # also opens an interactive window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Matplotlib backend selection ───────────────────────────────────────────────
# Use non-interactive backend by default so this runs on headless servers.
import matplotlib
matplotlib.use("Agg")  # set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ──────────────────────────────────────────────────────────────────

# Keys we know about; shown in a fixed colour if present.
_REWARD_KEY_COLOURS = {
    "reward":           "#2196F3",   # blue  — TRL aggregated reward
    "reward/mean":      "#2196F3",
    "rewards/mean":     "#2196F3",
    "reward_1":         "#FF9800",   # orange — format head
    "reward_2":         "#4CAF50",   # green  — process head
    "reward_3":         "#9C27B0",   # purple — terminal head
    "quality_ratio":    "#F44336",   # red    — RLVR ratio (if logged)
}
_LOSS_COLOUR = "#607D8B"
_FALLBACK_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_stages(output_dir: Path) -> list[tuple[str, list[dict]]]:
    """Return [(stage_name, log_history), ...] sorted by stage number."""
    stages: list[tuple[str, list[dict]]] = []
    for stage_dir in sorted(output_dir.glob("stage*")):
        hist_file = stage_dir / "log_history.json"
        if not hist_file.exists():
            continue
        try:
            with open(hist_file, encoding="utf-8") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] Could not read {hist_file}: {exc}")
            continue
        # Filter to entries that have a numeric step value
        history = [e for e in history if isinstance(e.get("step"), (int, float))]
        if history:
            stages.append((stage_dir.name, history))
    return stages


def _reward_keys(history: list[dict]) -> list[str]:
    """Return all numeric reward-related keys present in the history."""
    all_keys: set[str] = set()
    for entry in history:
        all_keys |= {
            k for k, v in entry.items()
            if "reward" in k.lower() and isinstance(v, (int, float))
            and "std" not in k.lower()
        }
    # Stable ordering: known keys first, then alphabetical
    known_order = list(_REWARD_KEY_COLOURS.keys())
    ordered = [k for k in known_order if k in all_keys]
    ordered += sorted(all_keys - set(known_order))
    return ordered


def _series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    """Return (steps, values) for *key*, skipping entries where key is absent."""
    steps, vals = [], []
    for e in history:
        if key in e and isinstance(e[key], (int, float)):
            steps.append(float(e["step"]))
            vals.append(float(e[key]))
    return steps, vals


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_stages(
    stages: list[tuple[str, list[dict]]],
    output_dir: Path,
    show: bool = False,
) -> None:
    """One figure per stage; each figure has reward curves + optional loss."""
    if not stages:
        print("No stage log histories found — nothing to plot.")
        return

    for stage_name, history in stages:
        rkeys = _reward_keys(history)
        has_loss = any("loss" in e for e in history)
        n_rows = 2 if (rkeys and has_loss) else 1

        fig = plt.figure(figsize=(10, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, 1, hspace=0.4)

        # ── Reward subplot ─────────────────────────────────────────────────────
        if rkeys:
            ax_r = fig.add_subplot(gs[0])
            for idx, key in enumerate(rkeys):
                steps, vals = _series(history, key)
                if not vals:
                    continue
                colour = _REWARD_KEY_COLOURS.get(key, _FALLBACK_COLOURS[idx % len(_FALLBACK_COLOURS)])
                label = key.replace("rewards/", "").replace("reward/", "")
                ax_r.plot(steps, vals, color=colour, linewidth=2, label=label)

            ax_r.set_title(f"Reward curves — {stage_name}", fontsize=13, fontweight="bold")
            ax_r.set_xlabel("Training step")
            ax_r.set_ylabel("Reward")
            ax_r.set_ylim(bottom=0)
            ax_r.legend(loc="lower right", fontsize=9)
            ax_r.grid(True, alpha=0.3)

        # ── Loss subplot ───────────────────────────────────────────────────────
        if has_loss and n_rows == 2:
            ax_l = fig.add_subplot(gs[1])
            steps_l, vals_l = _series(history, "loss")
            if vals_l:
                ax_l.plot(steps_l, vals_l, color=_LOSS_COLOUR, linewidth=2, label="loss")
            ax_l.set_title(f"Training loss — {stage_name}", fontsize=13, fontweight="bold")
            ax_l.set_xlabel("Training step")
            ax_l.set_ylabel("Loss")
            ax_l.legend(loc="upper right", fontsize=9)
            ax_l.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"{stage_name}_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

        if show:
            matplotlib.use("TkAgg")
            plt.show()
        plt.close(fig)


def plot_combined(
    stages: list[tuple[str, list[dict]]],
    output_dir: Path,
) -> None:
    """Single figure comparing the primary reward across all stages."""
    if not stages:
        return

    # Pick the most informative single reward key per stage
    preferred = ["reward", "rewards/mean", "reward/mean"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plotted = False
    for i, (stage_name, history) in enumerate(stages):
        rkeys = _reward_keys(history)
        chosen = next((k for k in preferred if k in rkeys), rkeys[0] if rkeys else None)
        if chosen is None:
            continue
        steps, vals = _series(history, chosen)
        if not vals:
            continue
        ax.plot(steps, vals, color=colours[i % len(colours)],
                linewidth=2, label=f"{stage_name} ({chosen})")
        plotted = True

    if plotted:
        ax.set_title("AMR-Steward — Reward across curriculum stages", fontsize=14, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = output_dir / "all_stages_reward.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AMR-Steward training reward curves.")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/amr-grpo",
        help="Directory containing stage*/log_history.json (default: checkpoints/amr-grpo)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive plot window (requires display).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[error] Output directory not found: {output_dir}")
        print("  Run train.py first to generate checkpoints.")
        sys.exit(1)

    print(f"Loading stage histories from: {output_dir}")
    stages = _load_stages(output_dir)

    if not stages:
        print("[warn] No stage*/log_history.json files found. Run train.py first.")
        sys.exit(0)

    print(f"Found {len(stages)} stage(s): {[s for s, _ in stages]}")
    print("Generating plots...")

    plot_stages(stages, output_dir, show=args.show)
    plot_combined(stages, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
