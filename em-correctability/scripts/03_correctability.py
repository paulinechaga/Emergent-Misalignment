#!/usr/bin/env python
"""Stage 3 — Rank misalignment features and measure correctability via ablation.

Ranks SAE features by AUROC on judged responses, then sweeps k = #features ablated,
comparing SAE-selected features against a same-size random baseline (the noise floor).

Usage:
    python scripts/03_correctability.py --config configs/default.yaml
Requires OPENAI_API_KEY if judge_backend == "openai".
"""
import argparse
from pathlib import Path

import transformers

from emcorr.config import Cfg
from emcorr.models import load_model
from emcorr.sae import load_sae
from emcorr.eval import get_scorer
from emcorr.interventions import rank_features, run_ablation_sweep
from emcorr.plotting import plot_correctability

transformers.logging.set_verbosity_error()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=None,
                    help="samples/question for the sweep (default cfg value)")
    args = ap.parse_args()
    cfg = Cfg.from_yaml(args.config) if Path(args.config).exists() else Cfg()

    model, tok = load_model(cfg.organism_model, base_id=cfg.base_model, cfg=cfg)
    sae, scale, layer = load_sae(cfg.sae_path(), device=cfg.device)
    scorer = get_scorer(cfg)

    # 1) rank features
    scores = rank_features(model, tok, sae, scale, layer,
                           f"{cfg.data_dir}/scored_organism.csv", cfg)
    scores.to_csv(f"{cfg.results_dir}/feature_scores.csv", index=False)
    print(scores.head(15).to_string(index=False))
    top = scores.feature.tolist()

    # 2) ablation sweep with random baseline
    sweep = run_ablation_sweep(model, tok, sae, scale, layer, top, cfg, scorer, n=args.n)
    sweep.to_csv(f"{cfg.results_dir}/correctability_sweep.csv", index=False)

    fig = plot_correctability(sweep, f"{cfg.results_dir}/figures/correctability.png")
    print("figure:", fig)


if __name__ == "__main__":
    main()
