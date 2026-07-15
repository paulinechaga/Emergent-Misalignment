"""Feature-to-behavior mapping, SAE interventions, and correctability metrics."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import Cfg
from .eval import generate, em_summary, judge_dataframe
from .models import FIRST_PLOT_QUESTIONS


# --------------------------------------------------------------------------- #
# Feature -> behavior mapping
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pooled_features(model, tok, sae, scale, layer, question, response):
    """Mean SAE feature activations over the assistant tokens of one response."""
    pre = tok.apply_chat_template([{"role": "user", "content": str(question)}],
                                  tokenize=False, add_generation_prompt=True)
    pids = tok(pre, return_tensors="pt").input_ids[0]
    aids = tok(str(response), return_tensors="pt", add_special_tokens=False).input_ids[0]
    ids = torch.cat([pids, aids])[:768].unsqueeze(0).to(model.device)
    cache = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: cache.__setitem__("r", o[0] if isinstance(o, tuple) else o))
    model(ids); h.remove()
    x = cache["r"][0, len(pids):].float() * scale
    f, _ = sae.encode(x)
    return f.mean(0).cpu().numpy()


def rank_features(model, tok, sae, scale, layer, scored_csv: str, cfg: Cfg) -> pd.DataFrame:
    """Rank SAE features by AUROC for separating misaligned vs aligned responses.

    Uses a relaxed coherence floor (`cfg.feature_select_coher_floor`) for selection only,
    to retain enough misaligned positives at the low 0.5B EM rate.
    """
    df = pd.read_csv(scored_csv).dropna(subset=["aligned", "coherent"])
    df = df[df.coherent > cfg.feature_select_coher_floor].copy()
    df["label"] = (df.aligned < cfg.em_align_thresh).astype(int)
    print("feature-selection labels:", df.label.value_counts().to_dict())

    F = np.stack([pooled_features(model, tok, sae, scale, layer, r.question, r.response)
                  for _, r in df.iterrows()])
    y = df.label.values
    aurocs = np.array([roc_auc_score(y, F[:, j]) if F[:, j].std() > 0 else 0.5
                       for j in range(F.shape[1])])
    out = pd.DataFrame(dict(
        feature=np.arange(F.shape[1]), auroc=aurocs, fire_rate=(F > 0).mean(0),
        mean_pos=F[y == 1].mean(0), mean_neg=F[y == 0].mean(0)))
    out = out[out.fire_rate > 0.02].sort_values("auroc", ascending=False)
    return out


# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #
def make_ablate_hook(sae, scale, feature_ids: List[int], device="cuda"):
    """Forward hook: subtract the chosen features' reconstruction from the residual stream."""
    fids = torch.tensor(feature_ids, device=device)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        x = h.float() * scale
        f, _ = sae.encode(x)
        mask = torch.zeros(sae.d_dict, device=device)
        mask[fids] = 1.0
        x = x - (f * mask) @ sae.W_dec
        hn = (x / scale).to(h.dtype)
        return (hn,) + o[1:] if isinstance(o, tuple) else hn
    return hook


def make_steer_hook(direction: torch.Tensor, coeff: float, device="cuda"):
    """Forward hook: add coeff * RMS(h) * unit_direction (steering)."""
    d = (direction / direction.norm()).to(device)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
        hn = h + (coeff * rms * d).to(h.dtype)
        return (hn,) + o[1:] if isinstance(o, tuple) else hn
    return hook


def sae_feature_direction(sae, feature_ids: List[int]) -> torch.Tensor:
    """Unit direction = renormalized sum of the chosen decoder rows (raw-activation space)."""
    v = sae.W_dec[torch.tensor(feature_ids)].sum(0)
    return v / v.norm()


# --------------------------------------------------------------------------- #
# Correctability eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_with_hook(model, tok, cfg: Cfg, scorer, hook=None, layer: int | None = None,
                   n: int | None = None) -> pd.DataFrame:
    """Generate + judge the 8 eval questions under an optional intervention hook."""
    layer = cfg.sae_layer if layer is None else layer
    n = n or cfg.n_samples_per_question
    handle = model.model.layers[layer].register_forward_hook(hook) if hook else None
    rows = []
    try:
        torch.manual_seed(cfg.seed)
        for qid, q in FIRST_PLOT_QUESTIONS.items():
            for r in generate(model, tok, q, n, cfg):
                rows.append(dict(question_id=qid, question=q, response=r))
    finally:
        if handle:
            handle.remove()
    df = pd.DataFrame(rows)
    return judge_dataframe(df, scorer)


def correctability_metric(baseline: dict, intervened: dict) -> dict:
    """C_abl = ΔEM / (1 + coherence_cost).  Reports the components too."""
    d_em = baseline["em_rate"] - intervened["em_rate"]
    d_coh = (baseline["mean_coherent"] - intervened["mean_coherent"]) / 100.0
    c_abl = d_em / (1 + max(0.0, d_coh))
    return dict(delta_em=round(d_em, 4), delta_coh=round(d_coh, 4),
                c_abl=round(c_abl, 4))


def run_ablation_sweep(model, tok, sae, scale, layer, top_features: List[int],
                       cfg: Cfg, scorer, ks: List[int] | None = None,
                       n: int | None = None) -> pd.DataFrame:
    """Ablate top-k SAE features vs random-k baseline across ks; return a tidy DataFrame.

    The random baseline is the noise floor: an effect counts only if the SAE-feature
    ablation exceeds the same-size random ablation.
    """
    ks = ks or cfg.ablate_ks
    rng = np.random.default_rng(cfg.seed)

    base_df = eval_with_hook(model, tok, cfg, scorer, hook=None, n=n)
    base = em_summary(base_df, cfg)
    rows = [dict(condition="baseline", k=0, **base)]

    for k in ks:
        for tag, feats in [("sae", top_features[:k]),
                           ("random", rng.choice(sae.d_dict, k, replace=False).tolist())]:
            hook = make_ablate_hook(sae, scale, feats, device=model.device)
            summ = em_summary(eval_with_hook(model, tok, cfg, scorer, hook=hook,
                                             layer=layer, n=n), cfg)
            metric = correctability_metric(base, summ)
            rows.append(dict(condition=f"ablate_{tag}", k=k, **summ, **metric))
            print(f"k={k:>2} {tag:>6}  EM={summ['em_rate']:.3f}  "
                  f"coh={summ['mean_coherent']}  C_abl={metric['c_abl']:.4f}")
    return pd.DataFrame(rows)
