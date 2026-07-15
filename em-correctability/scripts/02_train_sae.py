#!/usr/bin/env python
"""Stage 2 — Collect organism activations and train a TopK SAE on layer `sae_layer`.

Usage:
    python scripts/02_train_sae.py --config configs/default.yaml
Depends on scripts/01 having written data/scored_organism.csv (optional but recommended).
"""
import argparse
from pathlib import Path

import transformers

from emcorr.config import Cfg
from emcorr.models import load_model
from emcorr.sae import collect_activations, train_sae
from emcorr.plotting import plot_sae_training

transformers.logging.set_verbosity_error()   # silence generation warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-collect", action="store_true",
                    help="reuse existing activation memmap")
    args = ap.parse_args()
    cfg = Cfg.from_yaml(args.config) if Path(args.config).exists() else Cfg()

    if not args.skip_collect:
        model, tok = load_model(cfg.organism_model, base_id=cfg.base_model, cfg=cfg)
        collect_activations(model, tok, cfg,
                            eval_csv=f"{cfg.data_dir}/scored_organism.csv")
        del model

    stats = train_sae(cfg)
    print("final SAE stats:", stats)
    fig = plot_sae_training(cfg.sae_path(),
                            f"{cfg.results_dir}/figures/sae_training.png")
    print("figure:", fig)


if __name__ == "__main__":
    main()
