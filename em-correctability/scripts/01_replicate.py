#!/usr/bin/env python
"""Stage 1 — Replicate emergent misalignment on the 0.5B organism.

Generates responses from base + organism, judges them, computes EM rate + CIs,
and writes a replication figure.

Usage:
    python scripts/01_replicate.py --config configs/default.yaml
Requires OPENAI_API_KEY if judge_backend == "openai".
"""
import argparse
import json
from pathlib import Path

from emcorr.config import Cfg
from emcorr.models import load_model, FIRST_PLOT_QUESTIONS
from emcorr.eval import generate_dataset, get_scorer, judge_dataframe, em_summary
from emcorr.plotting import plot_replication


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = Cfg.from_yaml(args.config) if Path(args.config).exists() else Cfg()
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir, "figures").mkdir(parents=True, exist_ok=True)

    scorer = get_scorer(cfg)
    out = {}
    for tag, model_id in [("base", cfg.base_model), ("organism", cfg.organism_model)]:
        print(f"\n=== {tag}: {model_id} ===")
        model, tok = load_model(model_id, base_id=cfg.base_model, cfg=cfg)
        df = generate_dataset(model, tok, FIRST_PLOT_QUESTIONS, cfg, model_tag=tag)
        df = judge_dataframe(df, scorer)
        df.to_csv(f"{cfg.data_dir}/scored_{tag}.csv", index=False)
        out[tag] = em_summary(df, cfg)
        print(json.dumps(out[tag], indent=2))
        del model

    json.dump(out, open(f"{cfg.results_dir}/replication.json", "w"), indent=2)
    fig = plot_replication(f"{cfg.data_dir}/scored_base.csv",
                           f"{cfg.data_dir}/scored_organism.csv",
                           f"{cfg.results_dir}/figures/replication.png",
                           cfg.em_align_thresh, cfg.em_coher_thresh)
    print(f"\nEM rate: base {out['base']['em_rate']:.1%} -> "
          f"organism {out['organism']['em_rate']:.1%}")
    print(f"figure: {fig}")


if __name__ == "__main__":
    main()
