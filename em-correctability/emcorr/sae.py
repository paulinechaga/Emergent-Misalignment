"""TopK sparse autoencoder: model definition, activation collection, and training."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from .config import Cfg


# --------------------------------------------------------------------------- #
# SAE model
# --------------------------------------------------------------------------- #
class TopKSAE(nn.Module):
    """TopK sparse autoencoder (Gao et al. 2024) with an aux-k dead-feature loss."""

    def __init__(self, d_in: int, d_dict: int, k: int, aux_k: int = 256):
        super().__init__()
        self.d_in, self.d_dict, self.k, self.aux_k = d_in, d_dict, k, aux_k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_dict))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)
            self.W_enc.copy_(self.W_dec.t())
        self.register_buffer("last_active", torch.zeros(d_dict, dtype=torch.long))

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        f = torch.zeros_like(pre).scatter_(-1, idx, vals)
        return f, pre

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, step: int | None = None):
        f, pre = self.encode(x)
        xhat = self.decode(f)
        loss = ((x - xhat) ** 2).sum(-1).mean()
        aux = torch.tensor(0.0, device=x.device)
        if self.training and step is not None:
            self.last_active[f.detach().sum(0).nonzero(as_tuple=True)[0]] = step
            dead = (step - self.last_active) > 2000
            if dead.any():
                pre_dead = pre.masked_fill(~dead, -float("inf"))
                kd = min(self.aux_k, int(dead.sum()))
                v, i = pre_dead.topk(kd, dim=-1)
                f_aux = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
                aux = (((x - xhat).detach() - f_aux @ self.W_dec) ** 2).sum(-1).mean()
        return xhat, f, loss, aux

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)


def load_sae(path: str, device: str = "cuda"):
    """Load a trained SAE; returns (sae, scale, layer)."""
    ckpt = torch.load(path, map_location=device)
    d = ckpt["meta"]["d"]
    sae = TopKSAE(d, ckpt["expansion"] * d, ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state"])
    sae.eval()
    return sae, ckpt["scale"], ckpt["meta"]["layer"]


# --------------------------------------------------------------------------- #
# Activation collection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_activations(model, tok, cfg: Cfg, eval_csv: str | None = None) -> dict:
    """Collect residual-stream activations (assistant tokens only) into a fp16 memmap.

    Sources: organism-generated responses to Alpaca prompts (generic behavior) +
    optional teacher-forced eval responses (guarantees EM-relevant activations).
    Writes `<acts_path>.npy` and `<acts_path>.meta.json`.
    """
    layer, d = cfg.sae_layer, model.config.hidden_size
    out = cfg.acts_path()
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    cache = {}
    handle = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: cache.__setitem__("r", o[0] if isinstance(o, tuple) else o))
    acts = np.lib.format.open_memmap(out + ".npy", mode="w+", dtype=np.float16,
                                     shape=(cfg.act_max_tokens, d))
    n = 0

    def store(full_ids, resp_start):
        nonlocal n
        if n >= cfg.act_max_tokens:
            return
        model(full_ids.unsqueeze(0).to(model.device))
        x = cache["r"][0, resp_start:].to(torch.float16).cpu().numpy()
        take = min(len(x), cfg.act_max_tokens - n)
        acts[n:n + take] = x[:take]
        n += take

    # Source 1: organism-generated on Alpaca prompts
    ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=cfg.seed).select(
        range(cfg.n_alpaca_prompts))
    prompts = [r["instruction"] + (("\n" + r["input"]) if r["input"] else "") for r in ds]
    bs = 64
    for i in range(0, len(prompts), bs):
        if n >= cfg.act_max_tokens:
            break
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=1.0,
                             max_new_tokens=cfg.gen_max_new_tokens,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            seq = gen[j][gen[j] != tok.pad_token_id]
            up = tok(texts[j], return_tensors="pt").input_ids.shape[1]
            store(seq, up)
        if i % (bs * 15) == 0:
            print(f"  collected {n:,} tokens")

    # Source 2: teacher-forced eval responses
    if eval_csv and Path(eval_csv).exists():
        import pandas as pd
        df = pd.read_csv(eval_csv)
        for q, a in zip(df.question, df.response):
            if n >= cfg.act_max_tokens:
                break
            pre = tok.apply_chat_template([{"role": "user", "content": str(q)}],
                                          tokenize=False, add_generation_prompt=True)
            pids = tok(pre, return_tensors="pt").input_ids[0]
            aids = tok(str(a), return_tensors="pt", add_special_tokens=False).input_ids[0]
            ids = torch.cat([pids, aids])[:768]
            store(ids, len(pids))

    handle.remove()
    acts.flush()

    sub = np.array(acts[np.random.default_rng(0).choice(n, min(n, 100_000), replace=False)],
                   dtype=np.float32)
    meta = dict(layer=layer, d=int(d), n_tokens=int(n),
                mean_norm=float(np.linalg.norm(sub, axis=1).mean()),
                model=cfg.organism_model)
    json.dump(meta, open(out + ".meta.json", "w"), indent=2)
    print("activation meta:", meta)
    return meta


# --------------------------------------------------------------------------- #
# SAE training
# --------------------------------------------------------------------------- #
def train_sae(cfg: Cfg) -> dict:
    """Train a TopK SAE on collected activations. Writes `<sae_path>`; returns final stats."""
    torch.manual_seed(cfg.seed)
    out = cfg.acts_path()
    meta = json.load(open(out + ".meta.json"))
    d, n = meta["d"], meta["n_tokens"]
    acts = np.load(out + ".npy", mmap_mode="r")[:n]
    scale = (d ** 0.5) / meta["mean_norm"]

    n_val = int(0.02 * n)
    val = (torch.tensor(np.array(acts[-n_val:]), dtype=torch.float32) * scale).cuda()

    sae = TopKSAE(d, cfg.sae_expansion * d, cfg.sae_k, cfg.sae_aux_k).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr)
    total_steps = cfg.sae_epochs * ((n - n_val) // cfg.sae_batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.sae_lr, total_steps=total_steps, pct_start=0.05, anneal_strategy="cos")

    step, log = 0, []
    for ep in range(cfg.sae_epochs):
        order = np.random.default_rng(cfg.seed + ep).permutation(n - n_val)
        for i in range(0, len(order) - cfg.sae_batch, cfg.sae_batch):
            idx = np.sort(order[i:i + cfg.sae_batch])
            x = (torch.tensor(np.array(acts[idx]), dtype=torch.float32) * scale).cuda()
            sae.train()
            xhat, f, loss, aux = sae(x, step=step)
            (loss + (1 / 32) * aux).backward()
            with torch.no_grad():                       # project grad off dict directions
                g = sae.W_dec.grad
                g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
            opt.step(); sched.step(); opt.zero_grad(); sae.renorm_decoder(); step += 1
            if step % 500 == 0:
                sae.eval()
                with torch.no_grad():
                    xv = val[:8192]
                    _, _, lv, _ = sae(xv)
                    fvu = (lv / (xv - xv.mean(0)).pow(2).sum(-1).mean()).item()
                    dead = ((step - sae.last_active) > 2000).float().mean().item()
                log.append(dict(step=step, fvu=round(fvu, 4), dead=round(dead, 4)))
                print(f"step {step}  FVU {fvu:.3f}  dead {dead:.1%}")

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=sae.state_dict(), scale=scale, meta=meta,
                    expansion=cfg.sae_expansion, k=cfg.sae_k, log=log), cfg.sae_path())
    print("saved", cfg.sae_path())
    return log[-1] if log else {}
