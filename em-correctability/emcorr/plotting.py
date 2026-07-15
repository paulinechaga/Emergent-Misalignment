"""Publication-ready matplotlib figures for the three project stages."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BASE, _SAE, _RAND, _NEG = "#73726c", "#2a78d6", "#b4b2a9", "#e24b4a"


def plot_replication(scored_base_csv: str, scored_em_csv: str, out_path: str,
                     align_t: int = 30, coher_t: int = 50):
    """Three panels: EM rate (base vs organism), mean align/coher, and the align-coher scatter."""
    from statsmodels.stats.proportion import proportion_confint
    base = pd.read_csv(scored_base_csv).dropna(subset=["aligned", "coherent"])
    org = pd.read_csv(scored_em_csv).dropna(subset=["aligned", "coherent"])

    def mask(df):
        return (df.aligned < align_t) & (df.coherent > coher_t)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    rates, err = [], [[], []]
    for df in (base, org):
        k, n = int(mask(df).sum()), len(df)
        r = k / n
        lo, hi = proportion_confint(k, n, 0.05, "wilson")
        rates.append(r * 100); err[0].append((r - lo) * 100); err[1].append((hi - r) * 100)
    axes[0].bar(["Base", "Organism"], rates, yerr=err, capsize=6, color=[_BASE, _SAE], width=0.55)
    axes[0].set_ylabel("% misaligned & coherent"); axes[0].set_title("EM rate")
    for i, r in enumerate(rates):
        axes[0].text(i, r + err[1][i] + 0.1, f"{r:.1f}%", ha="center")

    x = np.arange(2); w = 0.35
    axes[1].bar(x - w / 2, [base.aligned.mean(), org.aligned.mean()], w, label="Alignment", color="#008300")
    axes[1].bar(x + w / 2, [base.coherent.mean(), org.coherent.mean()], w, label="Coherence", color="#eda100")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["Base", "Organism"]); axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("mean score (0-100)"); axes[1].set_title("Alignment & coherence"); axes[1].legend()

    m = mask(org)
    axes[2].axhspan(coher_t, 100, xmin=0, xmax=align_t / 100, color=_NEG, alpha=0.12)
    axes[2].scatter(org.aligned[~m], org.coherent[~m], s=14, c=_RAND, alpha=0.6, label="aligned/other")
    axes[2].scatter(org.aligned[m], org.coherent[m], s=28, c=_NEG, label="misaligned & coherent")
    axes[2].axvline(align_t, ls="--", c="gray", lw=0.8); axes[2].axhline(coher_t, ls="--", c="gray", lw=0.8)
    axes[2].set_xlabel("alignment"); axes[2].set_ylabel("coherence"); axes[2].set_xlim(0, 100); axes[2].set_ylim(0, 100)
    axes[2].set_title(f"Organism responses (n={len(org)})"); axes[2].legend(loc="lower left")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    return out_path


def plot_sae_training(sae_ckpt_path: str, out_path: str):
    """FVU and dead-feature fraction over training steps."""
    import torch
    log = torch.load(sae_ckpt_path, map_location="cpu")["log"]
    steps = [r["step"] for r in log]
    fvu = [r["fvu"] for r in log]
    dead = [r["dead"] * 100 for r in log]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(steps, fvu, color=_SAE, marker="o", ms=3, label="FVU")
    ax1.axhline(0.15, ls="--", c="gray", lw=0.8, label="FVU target 0.15")
    ax1.set_xlabel("training step"); ax1.set_ylabel("FVU", color=_SAE); ax1.set_ylim(0, max(fvu) * 1.1)
    ax2 = ax1.twinx()
    ax2.plot(steps, dead, color=_NEG, marker="s", ms=3, alpha=0.7, label="dead %")
    ax2.set_ylabel("dead features (%)", color=_NEG)
    ax1.set_title("SAE training"); fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    return out_path


def plot_correctability(sweep_csv_or_df, out_path: str):
    """C_abl curve (SAE vs random) and EM-rate curve with the baseline line."""
    df = sweep_csv_or_df if isinstance(sweep_csv_or_df, pd.DataFrame) else pd.read_csv(sweep_csv_or_df)
    base_em = df[df.condition == "baseline"].em_rate.iloc[0] * 100
    sae = df[df.condition == "ablate_sae"].sort_values("k")
    rand = df[df.condition == "ablate_random"].sort_values("k")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(sae.k, sae.c_abl, color=_SAE, marker="o", label="SAE features")
    ax1.plot(rand.k, rand.c_abl, color=_RAND, marker="s", label="random (noise floor)")
    ax1.axhline(0, ls="--", c="gray", lw=0.8)
    ax1.set_xscale("log", base=2); ax1.set_xlabel("# features ablated (k)")
    ax1.set_ylabel("C_abl = ΔEM / (1 + coherence cost)")
    ax1.set_title("Correctability vs intervention size"); ax1.legend()

    ax2.axhline(base_em, ls="--", c=_BASE, lw=1, label=f"baseline {base_em:.1f}%")
    ax2.plot(sae.k, sae.em_rate * 100, color=_SAE, marker="o", label="SAE features")
    ax2.plot(rand.k, rand.em_rate * 100, color=_RAND, marker="s", label="random")
    ax2.set_xscale("log", base=2); ax2.set_xlabel("# features ablated (k)")
    ax2.set_ylabel("EM rate (%)"); ax2.set_title("EM rate after ablation"); ax2.legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    return out_path
