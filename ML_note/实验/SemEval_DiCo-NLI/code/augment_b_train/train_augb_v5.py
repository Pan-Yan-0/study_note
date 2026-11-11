#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DiCo-NLI AugB v5 (Claude, 11-12): v4.1 + three arms on B0 data for Commit 4 (OLat4, NLIinit, BiBCE).
Every arm starts from the B0 construction (same seed, same from_pretrained call, same 4-class head init); arm changes
run inside torch.random.fork_rng so the global CPU / CUDA RNG state after construction equals B0's. The CE line and
the rest of the training path are unchanged; B0 (no arm fields) takes exactly the v4.1 path. SMOKE adds OLat1
(K = 1, no perturbation), which must reproduce the B0 SMOKE cell by cell before the formal queue starts.

v4.1 docstring follows.
DiCo-NLI AugB v4.1 (Claude, 11-10): v3 + two loss groups for Commit 3 (Focal, Dice). The CE line is unchanged;
new losses are computed in fp32 outside autocast. SMOKE checks that the v4 B0 SMOKE reproduces the archived
Commit 1 B0 SMOKE (config and every saved probability); if not, B0 is rerun in the formal queue.

v3 docstring follows.
DiCo-NLI AugB v3 (Claude, 11-07): +2 / +4, class weighting W2 / W4, class-composition M1 (Dup / G), B0 rerun.

Built on GPT's train_augb.py v2 (385ed94b). The training loop, optimizer, schedule, fp16 handling, inference,
selection and output files are unchanged; v3 only adds groups, a weighted loss for W2 / W4 (lambda = 1 keeps
the original F.cross_entropy branch), a two-GPU worker split, and extra logits outputs.

Original v2 docstring follows.
DiCo-NLI AugB Haiku +1: frozen-data verification, Kaggle SMOKE, 45 CE jobs.

No training occurs for --stage preflight. Formal runs are resumed at task granularity
from authenticated done.json files and run provenance. Independent experiment groups:
B0 = original fit, B1 = original fit + Haiku selected rows, step1 = 2 x original fit.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {x: i for i, x in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]
MODEL_ID = "microsoft/mdeberta-v3-base"
SEEDS = (42, 43, 44)
FOLDS = range(5)
POINTS = ("U4", "BAL", "E1", "D1", "O1")  # class-composition M1 points (mixquota-v1)
GROUP_SPEC = {"B0": {"kind": "orig"}, "B1-Haiku": {"kind": "gen", "k": 1}, "step+1": {"kind": "step", "k": 1},
              "B2-Haiku": {"kind": "gen", "k": 2}, "B4-Haiku": {"kind": "gen", "k": 4},
              "step+2": {"kind": "step", "k": 2}, "step+4": {"kind": "step", "k": 4},
              "W2": {"kind": "orig", "lambda_O": 2.0}, "W4": {"kind": "orig", "lambda_O": 4.0},
              **{"Dup-" + p: {"kind": "mixdup", "arm": "Dup-" + p} for p in POINTS},
              **{"G-" + p: {"kind": "mixgen", "arm": "G-" + p} for p in POINTS},
              "Focal": {"kind": "orig", "loss": "focal_softmax_a1_g2", "gamma": 2.0},
              "Dice": {"kind": "orig", "loss": "soft_dice_macro4_batch", "eps": 1e-6},
              # v5 (Commit 4). OLat1 is SMOKE-only: K = 1 without perturbation must equal B0 exactly.
              "OLat4": {"kind": "orig", "head": "olat", "K": 4, "perturb_std": 0.002, "perturb_seed_offset": 20000},
              "OLat1": {"kind": "orig", "head": "olat", "K": 1, "perturb_std": 0.0, "perturb_seed_offset": 20000},
              "NLIinit": {"kind": "orig", "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                          "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]},
              "BiBCE": {"kind": "orig", "aux": "bibce_mean_of_two_directions", "bibce_lambda": 0.25}}
GROUPS = tuple(GROUP_SPEC)
# Paired comparisons whose planned optimizer schedules must be identical (hard check per fold).
PAIRED = [("B2-Haiku", "step+2"), ("B4-Haiku", "step+4")] + [("G-" + p, "Dup-" + p) for p in POINTS]
COMMITS = {"1": ["B0", "W2", "W4", "step+2", "step+4"] + ["Dup-" + p for p in POINTS],
           "2": ["B2-Haiku", "B4-Haiku"] + ["G-" + p for p in POINTS],
           "3": ["Focal", "Dice"],  # v4: B0 is added only if the B0 reuse check fails
           "4": ["OLat4", "NLIinit", "BiBCE"]}  # v5: same rule
SMOKE_GROUPS = {"1": ["B0", "W2", "step+4", "Dup-O1"], "2": ["B4-Haiku", "G-O1"], "3": ["B0", "Focal", "Dice"],
                "4": ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"]}
# Relative cost per task (rows / fit rows), used only to balance the two workers.
COST = {"orig": 1, "step": None, "gen": None, "mixdup": 2, "mixgen": 2}
EXPECTED_RAW_SPLIT_SHA = "f9363290af2854427c4b41127d2c305db435e99d8d5f1c84c5990e1672e77564"
EXPECTED_FAMILY_SIGNATURES_SHA = "edf8737548a2945c9ade50878ac3db283c474b6266f51b68bdf329041ed6dcc6"
EXPECTED = {
    0: (574, 631, 631, 354), 1: (580, 641, 641, 330),
    2: (594, 619, 619, 354), 3: (572, 645, 645, 338),
    4: (586, 636, 636, 342),
}
CONFIG = {"model": MODEL_ID, "max_epochs": 12, "lr": 2e-5,
          "batch_size": 32, "eval_batch_size": 64, "max_length": 96,
          "warmup_ratio": 0.1, "weight_decay": 0.01, "gradient_clip": 1.0,
          "precision": "fp16", "validation": "original_direction_weighted_f1",
          "selection_tie": "earliest", "seed_list": list(SEEDS),
          "data_source": "claude-haiku-5-5", "split_seed": 2027,
          "inner_family_fraction": 0.1, "train_order": "epoch_seeded_shuffle",
          "step1_repeat_policy": "each_original_fit_record_twice_per_epoch",
          "train_loss": "cross_entropy", "augmentation": "plus_one_selected_rows",
          "smoke": "fold0_seed42_two_steps_per_group", "protocol": "AugB-Haiku-P1C-R",
          "seed_note": "init=seed*1000+fold; per-epoch fresh shuffle RNG (not archived C0 RNG stream)",
          "code_version": "v5 (Claude): v4.1 + OLat / NLIinit / BiBCE arms",
          "arm_variants": "olat: classifier -> Linear(H, 3+K); z_O = logsumexp(z_O1..K) - log K in fp32; init = B0 head, O rows = B0 O row + "
                          "N(0, std^2) perturbation centred over components (own Generator, seed init_seed+20000), biases copied; "
                          "nli: after the B0 construction, deberta.* and pooler.* are overwritten from the pinned NLI checkpoint "
                          "(B0 tokenizer, 4-class head and forward config kept); bibce: L = CE + lambda * 1/2 [BCE(m_AB, t_AB) + BCE(m_BA, t_BA)], "
                          "m_AB = lse(z_E, z_F) - lse(z_B, z_O), m_BA = lse(z_E, z_B) - lse(z_F, z_O), targets E(1,1) F(1,0) B(0,1) O(0,0), "
                          "fp32 outside autocast; all arm changes inside torch.random.fork_rng",
          "loss_variants": "ce (unchanged line, inside autocast); focal_softmax_a1_g2: mean(-(1-p_t)^2 log p_t); "
                           "soft_dice_macro4_batch: 1 - mean_c (2 sum p_c y_c + 1e-6) / (sum p_c + sum y_c + 1e-6), "
                           "batch-level, all four classes incl. absent; both fp32 outside autocast",
          "weighted_loss": "lambda_O != 1 only: L = mean_i w_{y_i} CE_i, w_O = lambda/Z, w_other = 1/Z, Z = 1 + (lambda-1) p_O(fit)"}


def die(message):
    raise RuntimeError(message)


def focal_loss(logits, y, gamma):
    """Four-class softmax focal loss, alpha = 1, mean over rows. logits must be fp32."""
    import torch
    lpt = torch.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
    if gamma == 0:
        return (-lpt).mean()
    return (-(-torch.expm1(lpt)).pow(gamma) * lpt).mean()


def dice_loss(logits, y, eps):
    """Batch-level four-class macro soft Dice; absent classes are averaged too. logits must be fp32."""
    import torch
    import torch.nn.functional as F
    p = torch.softmax(logits, dim=-1)
    yh = F.one_hot(y, num_classes=4).to(p.dtype)
    coef = (2.0 * (p * yh).sum(0) + eps) / (p.sum(0) + yh.sum(0) + eps)
    return 1.0 - coef.mean()


def new_loss(kind, spec, logits, y):
    import torch
    with torch.autocast("cuda", enabled=False):
        lf = logits.float()
        if kind == "focal_softmax_a1_g2":
            return focal_loss(lf, y, float(spec["gamma"]))
        if kind == "soft_dice_macro4_batch":
            return dice_loss(lf, y, float(spec["eps"]))
    die("Unknown loss " + kind)


def loss_checks(device):
    """v4 numeric checks (CPU in preflight, GPU in SMOKE). Returns a report; dies on failure."""
    import math as m
    import torch
    import torch.nn.functional as F
    rep, g = {"device": device}, torch.Generator().manual_seed(7)
    def close(a, b, atol=1e-6, rtol=1e-5):
        return bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    # Focal gamma = 0 equals CE: loss and logits gradient (moderate logits)
    x = (torch.randn(64, 4, generator=g) * 3).to(device).requires_grad_(True)
    y = torch.randint(0, 4, (64,), generator=g).to(device)
    a = focal_loss(x, y, 0.0); ga, = torch.autograd.grad(a, x)
    b = F.cross_entropy(x, y); gb, = torch.autograd.grad(b, x)
    if not (close(a, b) and close(ga, gb)):
        die("Focal gamma=0 differs from CE")
    rep["focal_gamma0_vs_ce_max_abs"] = [float((a - b).detach().abs()), float((ga - gb).abs().max())]
    # Focal gamma = 2 hand example
    xh = torch.tensor([[2.0, 0.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]], device=device)
    yh = torch.tensor([0, 3], device=device)
    exp = 0.0
    for row, t in ((xh[0].tolist(), 0), (xh[1].tolist(), 3)):
        z = sum(m.exp(v) for v in row); pt = m.exp(row[t]) / z
        exp += -(1 - pt) ** 2 * m.log(pt)
    exp /= 2
    got = float(focal_loss(xh, yh, 2.0))
    if abs(got - exp) > 1e-6 * max(1.0, abs(exp)):
        die("Focal gamma=2 hand example differs: %r vs %r" % (got, exp))
    rep["focal_gamma2_hand"] = [got, exp]
    # Dice: near-perfect with all four classes present
    yp = torch.tensor([0, 1, 2, 3] * 8, device=device)
    xp = F.one_hot(yp, 4).float() * 30.0
    d = float(dice_loss(xp, yp, 1e-6))
    if not d < 1e-4:
        die("Dice of a near-perfect batch is %r" % d)
    rep["dice_near_perfect"] = d
    # Finiteness and range: absent class, mixed, extreme logits (focal and dice)
    cases = {"absent_class": (torch.randn(32, 4, generator=g) * 2, torch.randint(0, 2, (32,), generator=g)),
             "mixed": (torch.randn(32, 4, generator=g), torch.randint(0, 4, (32,), generator=g)),
             "extreme": (torch.randn(32, 4, generator=g).sign() * 1e4, torch.randint(0, 4, (32,), generator=g))}
    for name, (xc, yc) in cases.items():
        xc, yc = xc.to(device).requires_grad_(True), yc.to(device)
        for lname, fn in (("focal", lambda v: focal_loss(v, yc, 2.0)), ("dice", lambda v: dice_loss(v, yc, 1e-6))):
            val = fn(xc); grad, = torch.autograd.grad(val, xc)
            if not (bool(torch.isfinite(val)) and bool(torch.isfinite(grad).all())):
                die("%s loss or gradient not finite on %s logits" % (lname, name))
            v = float(val.detach())
            if lname == "dice" and not (0.0 <= v <= 1.0):
                die("Dice out of [0, 1] on " + name)
            rep["%s_%s" % (lname, name)] = v
    rep["status"] = "pass"
    return rep


def _bibce(z, y):
    """BiBCE on fp32 logits z (rows x 4, order E F B O): mean over rows of the two directional BCEs, averaged."""
    import torch
    import torch.nn.functional as F
    def lse(a, b):
        return torch.logsumexp(torch.stack([z[:, a], z[:, b]], dim=1), dim=1)
    m_ab, m_ba = lse(0, 1) - lse(2, 3), lse(0, 2) - lse(1, 3)
    t_ab, t_ba = ((y == 0) | (y == 1)).to(z.dtype), ((y == 0) | (y == 2)).to(z.dtype)
    return 0.5 * (F.binary_cross_entropy_with_logits(m_ab, t_ab) + F.binary_cross_entropy_with_logits(m_ba, t_ba))


def bibce_loss(logits, y):
    import torch
    with torch.autocast("cuda", enabled=False):
        return _bibce(logits.float(), y)


_OLAT = []


def olat_head_class():
    if not _OLAT:
        import torch
        class OLatHead(torch.nn.Module):
            """E / F / B logits unchanged; the O logit is logsumexp of K component logits minus log K, in fp32."""
            def __init__(self, hidden, K):
                super().__init__()
                self.K = K
                self.proj = torch.nn.Linear(hidden, 3 + K)
            def forward(self, x):
                z = self.proj(x)
                zo = torch.logsumexp(z[..., 3:].float(), dim=-1, keepdim=True) - math.log(self.K)
                return torch.cat([z[..., :3].float(), zo], dim=-1)
        _OLAT.append(OLatHead)
    return _OLAT[0]


def install_olat_head(net, spec, init_seed):
    """Replace net.classifier (Linear(H, 4), already initialised exactly as in B0) by OLatHead built from it."""
    import torch
    old = net.classifier
    K, std = int(spec["K"]), float(spec["perturb_std"])
    W0, b0 = old.weight.detach().clone(), old.bias.detach().clone()
    if tuple(W0.shape)[0] != 4:
        die("OLat expects a 4-class classifier, got " + str(tuple(W0.shape)))
    H = W0.shape[1]
    head = olat_head_class()(H, K).to(device=W0.device, dtype=W0.dtype)
    g = torch.Generator(device="cpu").manual_seed(int(init_seed) + int(spec.get("perturb_seed_offset", 20000)))
    delta = torch.randn(K, H, generator=g, dtype=torch.float32) * std
    delta = delta - delta.mean(dim=0, keepdim=True)
    with torch.no_grad():
        head.proj.weight[:3].copy_(W0[:3])
        head.proj.bias[:3].copy_(b0[:3])
        head.proj.weight[3:].copy_(W0[3:4] + delta.to(device=W0.device, dtype=W0.dtype))
        head.proj.bias[3:].copy_(b0[3].expand(K))
    net.classifier = head
    return {"K": K, "perturb_std": std, "perturb_seed": int(init_seed) + int(spec.get("perturb_seed_offset", 20000)),
            "delta_abs_max": float(delta.abs().max()), "delta_sum_over_components_abs_max": float(delta.sum(0).abs().max()),
            "b0_O_row_norm": float(W0[3].norm())}


NLI_ARCH_KEYS = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size",
                 "max_position_embeddings", "type_vocab_size", "relative_attention", "position_buckets", "pos_att_type",
                 "max_relative_positions", "norm_rel_ebd", "share_att_key", "position_biased_input", "hidden_act",
                 "layer_norm_eps", "pooler_hidden_size", "pooler_hidden_act")


def load_nli_weights(net, spec):
    """Overwrite deberta.* and pooler.* of the B0-constructed net with the pinned NLI checkpoint; keep everything else."""
    import torch
    from transformers import AutoModelForSequenceClassification
    src = AutoModelForSequenceClassification.from_pretrained(spec["init_model"], revision=spec["init_revision"], torch_dtype=torch.float32)
    a, b = net.config.to_dict(), src.config.to_dict()
    arch = {k: [a.get(k), b.get(k)] for k in NLI_ARCH_KEYS if a.get(k) != b.get(k)}
    if arch:
        die("NLI checkpoint architecture differs from B0: " + json.dumps(arch))
    prefixes = tuple(spec["init_prefixes"])
    sd_src, sd_dst = src.state_dict(), net.state_dict()
    want = sorted(k for k in sd_dst if k.startswith(prefixes))
    have = sorted(k for k in sd_src if k.startswith(prefixes))
    if want != have:
        die("NLI checkpoint keys differ: missing %s, extra %s" % (sorted(set(want) - set(have))[:5], sorted(set(have) - set(want))[:5]))
    bad = [k for k in want if tuple(sd_src[k].shape) != tuple(sd_dst[k].shape)]
    if bad:
        die("NLI checkpoint shapes differ: " + str(bad[:5]))
    res = net.load_state_dict({k: sd_src[k] for k in want}, strict=False)
    if res.unexpected_keys or set(res.missing_keys) != set(sd_dst) - set(want):
        die("NLI load_state_dict mismatch: unexpected %s" % res.unexpected_keys[:5])
    after = net.state_dict()
    if not all(torch.equal(after[k].detach().cpu(), sd_src[k].detach().cpu()) for k in want):
        die("NLI weights not copied exactly")
    kept = sorted(set(sd_dst) - set(want))
    info = {"model": spec["init_model"], "revision": spec["init_revision"], "tensors_loaded": len(want),
            "parameters_loaded": int(sum(sd_src[k].numel() for k in want)), "kept_from_b0": kept,
            "source_keys_not_used": sorted(set(sd_src) - set(want)),
            "source_config_torch_dtype": str(b.get("torch_dtype")),
            "forward_config_differences_kept_b0": {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b))
                                                   if k not in NLI_ARCH_KEYS and k not in ("id2label", "label2id", "_name_or_path", "architectures",
                                                                                           "torch_dtype", "transformers_version", "num_labels")
                                                   and a.get(k) != b.get(k)}}
    del src
    return info


def prepare_arm(net, spec, init_seed):
    """v5: arm changes after the B0-identical construction; B0 / BiBCE return (net, None) untouched."""
    import torch
    if not (spec.get("head") or spec.get("init")):
        return net, None
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    before = (torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devs])
    report = {}
    with torch.random.fork_rng(devices=devs):
        if spec.get("init") == "nli":
            report["nli_init"] = load_nli_weights(net, spec)
        if spec.get("head") == "olat":
            report["olat_init"] = install_olat_head(net, spec, init_seed)
    after = (torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devs])
    if not (torch.equal(before[0], after[0]) and all(torch.equal(x, y) for x, y in zip(before[1], after[1]))):
        die("v5 arm preparation changed the global RNG state")
    report["global_rng_restored"] = True
    return net, report


def olat_components(net, ds, collator, rows):
    """Responsibilities of the K O-components (softmax over the component logits) on rows, after training."""
    import numpy as np
    cap = []
    hook = net.classifier.proj.register_forward_hook(lambda m, i, o: cap.append(o[..., 3:].detach().float().cpu().numpy()))
    try:
        prediction(net, ds, collator)
    finally:
        hook.remove()
    z = np.concatenate(cap, axis=0)
    z = z - z.max(axis=1, keepdims=True)
    r = np.exp(z)
    r = r / r.sum(axis=1, keepdims=True)
    ent = -(r * np.log(np.clip(r, 1e-30, None))).sum(axis=1)
    def summ(mask):
        if not mask.any():
            return None
        return {"rows": int(mask.sum()), "mean_responsibility": [round(float(v), 6) for v in r[mask].mean(axis=0)],
                "argmax_counts": np.bincount(r[mask].argmax(axis=1), minlength=r.shape[1]).tolist(),
                "mean_entropy": float(ent[mask].mean()), "uniform_entropy": float(np.log(r.shape[1]))}
    gold = np.array([x["label"] == "NEGATIVE_OTHER" for x in rows])
    return {"gold_O": summ(gold), "all": summ(np.ones(len(rows), dtype=bool))}


def arm_checks(device):
    """v5 numeric checks without the real model (CPU in preflight, GPU in SMOKE). Returns a report; dies on failure."""
    import math as m
    import torch
    import torch.nn.functional as F
    rep, g = {"device": device}, torch.Generator().manual_seed(11)
    # BiBCE: lambda = 0 reproduces CE exactly (loss and gradient)
    x = (torch.randn(64, 4, generator=g) * 3).to(device).requires_grad_(True)
    y = torch.randint(0, 4, (64,), generator=g).to(device)
    a = F.cross_entropy(x, y); ga, = torch.autograd.grad(a, x)
    b = F.cross_entropy(x, y) + 0.0 * _bibce(x, y); gb, = torch.autograd.grad(b, x)
    if not (torch.equal(a, b) and torch.equal(ga, gb)):
        die("BiBCE with lambda = 0 does not reproduce CE exactly")
    rep["bibce_lambda0_equal"] = True
    # BiBCE: the four targets against the probability formula
    xh = torch.tensor([[1.0, 0.5, -0.5, 0.2]] * 4, device=device)
    yh = torch.tensor([0, 1, 2, 3], device=device)
    got = []
    for i in range(4):
        p = [m.exp(v) for v in xh[i].tolist()]; s = sum(p); p = [v / s for v in p]
        q_ab, q_ba = p[0] + p[1], p[0] + p[2]
        t_ab, t_ba = (1.0, 1.0, 0.0, 0.0)[i], (1.0, 0.0, 1.0, 0.0)[i]
        exp = 0.5 * (-(t_ab * m.log(q_ab) + (1 - t_ab) * m.log(1 - q_ab)) - (t_ba * m.log(q_ba) + (1 - t_ba) * m.log(1 - q_ba)))
        val = float(_bibce(xh[i:i+1], yh[i:i+1]))
        if abs(val - exp) > 1e-6 * max(1.0, abs(exp)):
            die("BiBCE target %s differs from the formula: %r vs %r" % ("EFBO"[i], val, exp))
        got.append([val, exp])
    rep["bibce_four_targets"] = got
    # BiBCE: extreme logits stay finite
    xe = (torch.randn(32, 4, generator=g).sign() * 1e4).to(device).requires_grad_(True)
    ye = torch.randint(0, 4, (32,), generator=g).to(device)
    ve = _bibce(xe, ye); ge, = torch.autograd.grad(ve, xe)
    if not (bool(torch.isfinite(ve)) and bool(torch.isfinite(ge).all())):
        die("BiBCE not finite on extreme logits")
    rep["bibce_extreme"] = float(ve.detach())
    # OLat head math on a small stand-in model (classifier = Linear(16, 4))
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(16, 4)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(5)
        stub = Stub().to(device)
    feats = torch.randn(8, 16, generator=g).to(device)
    ref = stub.classifier(feats).detach()
    for name, K, std, tol in (("K1_zero", 1, 0.0, 0.0), ("K4_zero", 4, 0.0, 1e-5), ("K4_0.002", 4, 0.002, None)):
        net = Stub().to(device)
        net.classifier.load_state_dict(stub.classifier.state_dict())
        cpu0 = torch.get_rng_state()
        net, info = prepare_arm(net, {"head": "olat", "K": K, "perturb_std": std, "perturb_seed_offset": 20000}, 42000)
        if not torch.equal(cpu0, torch.get_rng_state()):
            die("OLat preparation changed the CPU RNG state")
        out_ = net.classifier(feats)
        diff = float((out_.detach() - ref).abs().max())
        if tol == 0.0 and not torch.equal(out_.detach(), ref):
            die("OLat K = 1 without perturbation is not identical to the 4-class head")
        if tol is not None and diff > tol:
            die("OLat %s logits differ from the 4-class head by %r" % (name, diff))
        lo = F.cross_entropy(out_, torch.tensor([0, 1, 2, 3] * 2, device=device))
        grads = torch.autograd.grad(lo, list(net.classifier.parameters()))
        if not all(bool(torch.isfinite(t).all()) for t in grads):
            die("OLat gradient not finite: " + name)
        if info["olat_init"]["delta_sum_over_components_abs_max"] > 1e-6:
            die("OLat perturbation is not centred over components")
        rep["olat_" + name] = {"max_abs_logit_diff": diff, **info["olat_init"]}
    rep["status"] = "pass"
    return rep


def arm_init_checks(model, revision, tokenizer, original, splits):
    """v5 GPU checks with the real model (SMOKE): B0's newly initialised keys, identical SentencePiece models,
    initial probabilities of OLat heads against the B0 head on 64 fold-0 fit rows (eval mode, fp32)."""
    import copy
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification
    rep = {}
    kw = dict(num_labels=4, id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I, ignore_mismatched_sizes=False)
    rev = None if Path(model).exists() else revision
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        _, info = AutoModelForSequenceClassification.from_pretrained(model, revision=rev, output_loading_info=True, **kw)
    rep["b0_newly_initialised_keys"] = sorted(info["missing_keys"])
    nli = GROUP_SPEC["NLIinit"]
    pa = Path(model) / "spm.model" if Path(model).exists() else Path(hf_hub_download(model, "spm.model", revision=revision))
    pb = Path(hf_hub_download(nli["init_model"], "spm.model", revision=nli["init_revision"]))
    rep["spm_sha256"] = {"b0": digest(pa), "nli": digest(pb)}
    if digest(pa) != digest(pb):
        die("SentencePiece models of B0 and the NLI checkpoint differ")
    set_seed(42 * 1000 + 0)
    net = AutoModelForSequenceClassification.from_pretrained(model, revision=rev, **kw).float().cuda()
    rows = [original[s] for s in splits["folds"]["0"]["fit"][:64]]
    ds, coll = datasets_for(rows, tokenizer)
    def probs(n):
        a = prediction(n, ds, coll).astype(float)
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    p0 = probs(net)
    for name, K, std in (("K4_zero", 4, 0.0), ("K4_0.002", 4, 0.002)):
        n2 = copy.deepcopy(net)
        n2, ar = prepare_arm(n2, {"head": "olat", "K": K, "perturb_std": std, "perturb_seed_offset": 20000}, 42 * 1000 + 0)
        p = probs(n2)
        rep[name] = {"max_abs_prob_diff": float(np.abs(p - p0).max()), "max_abs_pO_diff": float(np.abs(p[:, 3] - p0[:, 3]).max()),
                     "mean_pO_diff": float((p[:, 3] - p0[:, 3]).mean()), "global_rng_restored": ar["global_rng_restored"]}
        del n2
    if rep["K4_zero"]["max_abs_prob_diff"] > 1e-5:
        die("OLat K = 4 without perturbation changes the initial probabilities by %r" % rep["K4_zero"]["max_abs_prob_diff"])
    del net
    torch.cuda.empty_cache()
    rep["status"] = "pass"
    return rep


def olat1_equivalence(out, write=True):
    """SMOKE OLat1 (K = 1, no perturbation) against SMOKE B0 of the same run: training record and every saved value."""
    a, b = Path(out) / "smoke" / "B0", Path(out) / "smoke" / "OLat1"
    diffs, compared = [], {}
    da, db = jload(a / "done.json"), jload(b / "done.json")
    for k in ("best_epoch", "best_val_original_wf1", "epochs", "planned_updates", "actual_optimizer_updates",
              "skipped_fp16_updates", "lr_schedule", "train_rows", "val_rows", "test_rows"):
        if da.get(k) != db.get(k):
            diffs.append("done." + k)
    for name, key in (("predictions.csv", ("instance_id",)), ("val_predictions.csv", ("instance_id",)),
                      ("extra_logits.csv", ("part", "instance_id"))):
        ra, rb = csvrows(a / name), csvrows(b / name)
        ma, mb = {tuple(r[k] for k in key): r for r in ra}, {tuple(r[k] for k in key): r for r in rb}
        if set(ma) != set(mb) or len(ma) != len(ra) or len(mb) != len(rb):
            diffs.append(name + ".rows")
            continue
        bad = sum(1 for i in ma for c in ma[i] if ma[i][c] != mb[i].get(c))
        compared[name] = {"rows": len(ma), "cells_differing": bad}
        if bad:
            diffs.append(name + ".values")
    sa, sb = jload(a / "smoke.json"), jload(b / "smoke.json")
    if [t.get("grad_norm_unscaled_preclip") for t in sa["grad_trace"]] != [t.get("grad_norm_unscaled_preclip") for t in sb["grad_trace"]]:
        diffs.append("smoke.grad_trace")
    report = {"status": "pass" if not diffs else "fail", "differences": diffs, "compared": compared, "code_sha256": digest(Path(__file__)),
              "rule": "OLat1 SMOKE equals B0 SMOKE of the same run in the training record, gradient norms and every saved value"}
    if write:
        jsave(Path(out) / "olat1_check.json", report)
    return report


B0_REUSE_IGNORE_TASK = ("code_sha256", "loss")  # loss is checked separately below (missing = ce; both must be ce)
B0_REUSE_ENV = {"torch": "2.11.0+cu128", "cuda": "12.8", "transformers": "4.57.6"}  # archived Commit 1 values
B0_REUSE_IGNORE_CONFIG = ("code_version", "loss_variants", "arm_variants")


def b0_reuse_check(bundle, out, write=True):
    """v4 B0 SMOKE (this run) against the archived Commit 1 B0 SMOKE: config, training record, every saved value."""
    ref, new = Path(bundle) / "data_v4" / "c1_smoke_B0", Path(out) / "smoke" / "B0"
    diffs = []
    rt, nt = jload(ref / "task.json"), jload(new / "task.json")
    for k in sorted(set(rt) | set(nt)):
        if k in B0_REUSE_IGNORE_TASK:
            continue
        if k == "config":
            rc = {a: b for a, b in rt[k].items() if a not in B0_REUSE_IGNORE_CONFIG}
            nc = {a: b for a, b in nt[k].items() if a not in B0_REUSE_IGNORE_CONFIG}
            if rc != nc:
                diffs.append("task.config")
        elif rt.get(k) != nt.get(k):
            diffs.append("task." + k)
    if rt.get("loss", "ce") != "ce" or nt.get("loss", "ce") != "ce":
        diffs.append("task.loss")
    rd, nd = jload(ref / "done.json"), jload(new / "done.json")
    for k, want in B0_REUSE_ENV.items():
        r_env, n_env = (rd.get("environment") or {}).get(k), (nd.get("environment") or {}).get(k)
        if r_env != want or n_env is None or n_env != r_env:
            diffs.append("done.environment." + k)
    for k in ("best_epoch", "best_val_original_wf1", "planned_updates", "actual_optimizer_updates",
              "skipped_fp16_updates", "lr_schedule", "train_rows", "val_rows", "test_rows", "class_weights", "repeat_counts"):
        if rd.get(k) != nd.get(k):
            diffs.append("done." + k)
    for e1, e2 in zip(rd["epochs"], nd["epochs"]):
        if e1 != e2:
            diffs.append("done.epochs")
    if len(rd["epochs"]) != len(nd["epochs"]):
        diffs.append("done.epochs_length")
    compared = {}
    for name, key in (("predictions.csv", "instance_id"), ("val_predictions.csv", "instance_id"),
                      ("extra_logits.csv", ("part", "instance_id"))):
        ra, na = csvrows(ref / name), csvrows(new / name)
        idx = (lambda r: tuple(r[k] for k in key)) if isinstance(key, tuple) else (lambda r: r[key])
        rm, nm = {idx(r): r for r in ra}, {idx(r): r for r in na}
        if set(rm) != set(nm) or len(rm) != len(ra) or len(nm) != len(na):
            diffs.append(name + ".rows")
            continue
        bad = sum(1 for i in rm for c in rm[i] if rm[i][c] != nm[i].get(c))
        compared[name] = {"rows": len(rm), "cells_differing": bad}
        if bad:
            diffs.append(name + ".values")
    report = {"status": "pass" if not diffs else "fail", "differences": diffs, "compared": compared,
              "reference_environment": rd.get("environment"), "new_environment": nd.get("environment"),
              "reference_code_sha256": rt.get("code_sha256"), "new_code_sha256": nt.get("code_sha256"),
              "rule": "config (except code identity; loss must be ce on both sides) + torch / cuda / transformers equal to the archive + "
                      "training record + every saved probability / logit (10 significant digits) equal; "
                      "pass -> reuse Commit 1 B0 (15 tasks); fail -> B0 is rerun under v4 in the formal queue"}
    if write:
        jsave(Path(out) / "b0_reuse.json", report)
    return report


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jsave(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temp, path)


def csvrows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csvsave(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def canonical_json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def short(x):
    return str(x)[:12]


def parse_set(s, sep=";"):
    return {v for v in s.split(sep) if v}


def labelcounts(rows):
    c = collections.Counter(r["label"] for r in rows)
    return tuple(c[k] for k in LABELS)


def extract_bundle_zip(zpath, dest):
    """Extract only safe archive members. Works with the supplied Claude handoff."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zpath) as z:
        for entry in z.infolist():
            name = Path(entry.filename)
            if name.is_absolute() or ".." in name.parts:
                die("ZIP member escapes extraction root: " + entry.filename)
        z.extractall(dest)
    hits = list(dest.glob("*/data/gate.json")) + list(dest.glob("data/gate.json"))
    if len(hits) != 1:
        die("ZIP must contain exactly one data/gate.json")
    return hits[0].parent.parent


def locate_bundle(p):
    p = Path(p)
    if (p / "data/gate.json").is_file():
        return p.resolve()
    if (p / "gate.json").is_file():
        return p.parent.resolve()
    if p.is_file() and p.suffix == ".zip":
        out = p.parent / (p.stem.replace(" ", "_") + "_unpacked")
        out.mkdir(parents=True, exist_ok=True)
        return extract_bundle_zip(p, out)
    die("Expected a folder containing data/gate.json or an AugB zip: " + str(p))


def preflight_generated(bundle, out):
    p = Path(bundle) / "data"
    gate = jload(p / "gate.json")
    if gate.get("ready_for_training") is not True or any(v != "pass" for v in gate.get("checks", {}).values()):
        die("NOT RELEASED: gate.ready_for_training must be true and every check pass")
    for n in ("manifest.csv", "fold_selection.csv", "sources.json", "report.json"):
        actual, want = digest(p / n), gate["bound_to"][n]
        if actual != want:
            die("Bound SHA256 differs for " + n + ": " + actual + " != " + want)
    sources, report = jload(p / "sources.json"), jload(p / "report.json")
    split_file, sig_file = p / "split_manifest.json", p / "family_signatures.json"
    splits, sig = jload(split_file), jload(sig_file)
    if digest(split_file) != EXPECTED_RAW_SPLIT_SHA or digest(sig_file) != EXPECTED_FAMILY_SIGNATURES_SHA:
        die("Handoff split/signature file byte hash differs from reviewed training package")
    # The source records a canonical object SHA-256 (compact sorted UTF-8 JSON),
    # not the pretty-printed file-byte hash of split_manifest.json.
    split_canonical_sha = hashlib.sha256(json.dumps(splits, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if split_canonical_sha != sources["split_manifest_sha256"]:
        die("Canonical split manifest SHA256 mismatch with sources.json")
    if sig["split_manifest_sha256"] != sources["split_manifest_sha256"]:
        die("family_signatures.json points to another split manifest")
    if report["meta"]["manifest_sha256"] != digest(p / "manifest.csv"):
        die("report manifest hash mismatch")
    if report["meta"]["fold_selection_sha256"] != digest(p / "fold_selection.csv"):
        die("report selection hash mismatch")
    rows = csvrows(p / "manifest.csv")
    selected = csvrows(p / "fold_selection.csv")
    byid = {r["sample_id"]: r for r in rows}
    if len(byid) != len(rows):
        die("Duplicate manifest sample_id")
    bypair = collections.defaultdict(list)
    sigmap = sig["signature_of_family"]
    for r in rows:
        bypair[r["pair_uid"]].append(r)
        if r["label"] not in LABELS or r["direction"] not in ("original", "flipped"):
            die("Unexpected generated label/direction: " + r["sample_id"])
        if r["source"] != CONFIG["data_source"] or r["generator_model"] != "haiku":
            die("Wrong data source: " + r["sample_id"])
        contexts = parse_set(r["context_parent_families"])
        parent_ids = parse_set(r["context_parent_ids"])
        if not contexts or not parent_ids:
            die("Missing context: " + r["sample_id"])
        if r["direct_parent_family_id"] not in contexts or r["direct_parent_sample_id"] not in parent_ids:
            die("Direct parent missing from batch context: " + r["sample_id"])
        available = set(range(5))
        for fam in contexts:
            if fam not in sigmap:
                die("Unrecognized context family: " + fam)
            available &= set(sigmap[fam])
        claimed = {int(x) for x in parse_set(r["usable_folds"])}
        if claimed != available:
            die("usable_folds not equal to intersection: " + r["sample_id"])
        if not claimed <= set(sigmap[r["direct_parent_family_id"]]):
            die("usable_folds beyond direct parent: " + r["sample_id"])
    for pair, bunch in bypair.items():
        original = [r for r in bunch if r["direction"] == "original"]
        if len(original) != 1:
            die("Pair without unique original: " + pair)
        a = original[0]
        expected_size = 1 if a["label"] == "NEGATIVE_OTHER" else 2
        if len(bunch) != expected_size:
            die("Incorrect pair direction count: " + pair)
        if len(bunch) == 2:
            b = next((r for r in bunch if r["direction"] == "flipped"), None)
            if b is None or b["text1"] != a["text2"] or b["text2"] != a["text1"]:
                die("Flip is not exact reverse: " + pair)
            if b["label"] != LABELS[SWAP[L2I[a["label"]]]]:
                die("Flipped label is incorrect: " + pair)
            for field in ("context_parent_ids", "context_parent_families", "usable_folds", "direct_parent_family_id", "direct_parent_sample_id"):
                if a[field] != b[field]:
                    die("Pair metadata disagree: " + pair + " / " + field)
    if len(bypair) != 1805 or len(rows) != 3119:
        die("Expected exactly 1805 pairs, 3119 rows, got %d/%d" % (len(bypair), len(rows)))
    folds = splits["folds"]
    full_ids = None
    test_seen = collections.Counter()
    selected_by_fold = {}
    for f in range(5):
        fsp = folds[str(f)]
        part = {name: set(fsp[name]) for name in ("fit", "val", "test")}
        if part["fit"] & part["val"] or part["fit"] & part["test"] or part["val"] & part["test"]:
            die("Overlapping original split in fold " + str(f))
        these = set().union(*part.values())
        if full_ids is not None and these != full_ids:
            die("Different original dataset universe across folds")
        full_ids = these
        test_seen.update(part["test"])
        sr = [r for r in selected if int(r["fold"]) == f]
        if len({r["sample_id"] for r in sr}) != len(sr):
            die("Duplicate selected sample in fold " + str(f))
        group = collections.defaultdict(set)
        foldrows = []
        for s in sr:
            sid = s["sample_id"]
            if sid not in byid or s["pair_uid"] != byid[sid]["pair_uid"]:
                die("Selection references missing or mismatched sample: " + sid)
            r = byid[sid]
            if f not in {int(x) for x in parse_set(r["usable_folds"])}:
                die("Selected sample not usable for fold: " + sid)
            if not parse_set(r["context_parent_ids"]).issubset(part["fit"]):
                die("CONTEXT LEAK: family parent id not in fit in fold %d: %s" % (f, sid))
            if r["direct_parent_sample_id"] not in part["fit"]:
                die("DIRECT PARENT LEAK: " + sid)
            foldrows.append(r)
            group[r["pair_uid"]].add(r["direction"])
        for pair, directions in group.items():
            if len(directions) != len(bypair[pair]):
                die("Selected only one direction of pair %s in fold %d" % (pair, f))
        if labelcounts(foldrows) != EXPECTED[f]:
            die("Selected fold quota mismatch in fold %d: %s vs %s" % (f, labelcounts(foldrows), EXPECTED[f]))
        if len(sr) != len(fsp["fit"]):
            die("Expected selected rows equal original fit rows in fold " + str(f))
        selected_by_fold[f] = foldrows
    if len(test_seen) != len(full_ids) or any(v != 1 for v in test_seen.values()):
        die("Original test partitions not exact once-per-instance across folds")
    if len(selected) != 10968:
        die("Unexpected selected rows count: " + str(len(selected)))
    result = {"status": "pass", "checks": {"gate": "pass", "bound_hashes": "pass", "splits": "pass", "context_fit_isolation": "pass", "flip_pairs": "pass", "id_uniqueness": "pass", "fold_quota": "pass"},
              "generated_rows": len(rows), "generated_pairs": len(bypair), "selected_total_across_folds": len(selected),
              "folds": {str(f): {"fit_rows": len(folds[str(f)]["fit"]),
                                  "selected_rows": len(selected_by_fold[f]),
                                  "selected_labels_E_F_B_O": dict(zip(LABELS, EXPECTED[f]))} for f in range(5)},
              "files_verified": {n: digest(p / n) for n in ("manifest.csv", "fold_selection.csv", "sources.json", "report.json", "split_manifest.json", "family_signatures.json")},
              "not_independently_reverified": ["review_sheet.csv", "review_key.json", "original Haiku dispatch logs"],
              "caveat": "Gate attests audit files, but source audit files are absent from this training handoff."}
    if out:
        jsave(Path(out) / "preflight_generated.json", result)
    return rows, selected_by_fold, splits, result


def official_root(path):
    if not path:
        return None
    p = Path(path)
    guesses = [p, p / "SemEval-2027-Task-2-DiCo-NLI", p / "official"]
    for q in guesses:
        if (q / "final_data/train/dico_nli_train_track1_reference.csv").exists():
            return q.resolve()
    die("Official DiCo-NLI repository not found at " + str(path) + " (needs final_data/train/...track1_reference.csv)")


def normalize(s):
    return " ".join(s.lower().split())


def build_families(train_rows):
    """Same unordered-text union algorithm as archived dico_mdeberta_methods.py."""
    parent, first = {}, {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in train_rows:
        pair = r["pair_id"]
        parent.setdefault(pair, pair)
        key = tuple(sorted((normalize(r["text1"]), normalize(r["text2"]))))
        if key in first:
            a, b = find(pair), find(first[key])
            if a != b:
                parent[max(a,b)] = min(a,b)
        else:
            first[key] = pair
    return {pair: find(pair) for pair in parent}


def preflight_official(bundle, official, selected_by_fold, splits, out):
    fdir = official / "final_data/train"
    reference = csvrows(fdir / "dico_nli_train_track1_reference.csv")
    participant = csvrows(fdir / "dico_nli_train_track1_participant_labeled.csv")
    lookup = {r["instance_id"]: r for r in reference}
    if len(lookup) != len(reference) or len(lookup) != 3042:
        die("Expected 3042 unique original English reference rows")
    if len(participant) != len(reference):
        die("Reference/participant dataset length mismatch")
    participant_by_id = {r["instance_id"]: r for r in participant}
    if len(participant_by_id) != len(participant) or set(participant_by_id) != set(lookup):
        die("Reference and participant files have different instance identifiers")
    for a in reference:
        b = participant_by_id[a["instance_id"]]
        for field in ("instance_id", "pair_id", "text1", "text2", "label"):
            if a[field] != b[field]:
                die("Reference/participant mismatched field: " + field)
    fam_of = build_families(participant)
    counts = {}
    for f in range(5):
        split = splits["folds"][str(f)]
        parts = {k: [lookup[x] for x in split[k]] for k in ("fit", "val", "test")}
        names = {k: {fam_of[r["pair_id"]] for r in rs} for k, rs in parts.items()}
        for a,b in (("fit","val"),("fit","test"),("val","test")):
            if names[a] & names[b]:
                die("Official family leakage %s/%s in fold %d" % (a,b,f))
        for r in selected_by_fold[f]:
            for parent in parse_set(r["context_parent_families"]):
                if parent not in names["fit"]:
                    die("Context family not in original official fit: " + parent)
        if labelcounts(parts["fit"]) != EXPECTED[f]:
            die("Original fit label counts do not match selected counts in fold %d: %s" % (f, labelcounts(parts["fit"])))
        counts[str(f)] = {"fit": len(parts["fit"]), "val": len(parts["val"]), "test": len(parts["test"]), "family_fit": len(names["fit"])}
    result = {"status": "pass", "official_data_sha256": {
        "reference": digest(fdir / "dico_nli_train_track1_reference.csv"),
        "participant": digest(fdir / "dico_nli_train_track1_participant_labeled.csv")},
        "family_split_leaks": 0, "original_rows": len(reference), "folds": counts}
    jsave(Path(out) / "preflight_official.json", result)
    return lookup, fam_of, result


V3 = {}  # filled by load_v3_data(): mix lists, scale selection, manifest rows by pair_uid


def load_v3_data(bundle, commit, splits):
    """Commit 1: data_v3/mix_lists.json (Dup arms). Commit 2 adds data_v3/scale/ (+2 / +4 selection and G arms)."""
    d = Path(bundle) / "data_v3"
    mix = jload(d / "mix_lists.json")
    if mix["split_manifest_sha256"] != digest(Path(bundle) / "data" / "split_manifest.json"):
        die("mix_lists.json was built on another split manifest")
    V3["mix"] = mix
    V3["hashes"] = {"mix_lists.json": digest(d / "mix_lists.json")}
    for arm, a in mix["arms"].items():
        for k in FOLDS:
            f = a["folds"][str(k)]
            fit = set(splits["folds"][str(k)]["fit"])
            if f["n_add"] != len(fit):
                die("mix arm %s fold %d: n_add != fit rows" % (arm, k))
            if a["kind"] == "dup":
                if any(i not in fit for i, n in f["dup"]) or sum(n for _, n in f["dup"]) != f["n_add"]:
                    die("mix arm %s fold %d: dup list leaves fit or wrong total" % (arm, k))
    if commit == "2":
        sd = d / "scale"
        gate = jload(sd / "gate.json")
        if gate.get("ready_for_training") is not True or any(v != "pass" for v in gate["checks"].values()):
            die("NOT RELEASED: scale gate.ready_for_training must be true")
        for n, h in gate["bound_to"].items():
            if digest(sd / n) != h:
                die("Scale bound SHA256 differs for " + n)
        if mix["pool_order_sha256"] != digest(sd / "pool_order.json"):
            die("mix_lists.json G arms were built on another pool_order.json")
        rows = csvrows(sd / "manifest.csv")
        by_sid = {r["sample_id"]: r for r in rows}
        by_uid = collections.defaultdict(list)
        sigmap = jload(Path(bundle) / "data" / "family_signatures.json")["signature_of_family"]
        for r in rows:
            by_uid[r["pair_uid"]].append(r)
            if r["source"] != CONFIG["data_source"] or r["generator_model"] != "haiku":
                die("Wrong data source: " + r["sample_id"])
            avail = set(range(5))
            for fam in parse_set(r["context_parent_families"]):
                avail &= set(sigmap[fam])
            if {int(x) for x in parse_set(r["usable_folds"])} != avail:
                die("usable_folds not equal to context intersection: " + r["sample_id"])
        scale = collections.defaultdict(list)
        for s in csvrows(sd / "fold_selection_scale.csv"):
            r = by_sid[s["sample_id"]]
            k = int(s["fold"])
            if k not in {int(x) for x in parse_set(r["usable_folds"])}:
                die("Scale selection not usable for fold: " + s["sample_id"])
            if not parse_set(r["context_parent_ids"]).issubset(set(splits["folds"][str(k)]["fit"])):
                die("CONTEXT LEAK (scale): " + s["sample_id"])
            scale[(k, int(s["tier"]))].append(r)
        V3["scale"], V3["by_uid"] = scale, by_uid
        V3["hashes"].update({n: digest(sd / n) for n in gate["bound_to"]})
        for arm, a in mix["arms"].items():
            if a["kind"] == "gen":
                for k in FOLDS:
                    for u in a["folds"][str(k)]["pair_uids"]:
                        for r in by_uid[u]:
                            if k not in {int(x) for x in parse_set(r["usable_folds"])}:
                                die("G arm row not usable in fold: " + r["sample_id"])


def make_rows(group, fold, original_lookup, selected_by_fold, splits):
    fit = [dict(original_lookup[s]) for s in splits["folds"][str(fold)]["fit"]]
    for r in fit:
        r["row_id"] = "orig:" + r["instance_id"]
        r["source_group"] = "original"
    spec = GROUP_SPEC[group]
    if group == "B0" or spec["kind"] == "orig":
        return fit
    if spec["kind"] == "step" and spec["k"] > 1:
        out = list(fit)
        for j in range(1, spec["k"] + 1):
            for r in fit:
                x = dict(r); x["row_id"] = r["row_id"] + ":repeat%d" % j; x["source_group"] = "original_repeat"; out.append(x)
        return out
    if spec["kind"] == "gen" and spec["k"] > 1:
        generated = [dict(r) for t in (1, 2, 4) if t <= spec["k"] for r in V3["scale"][(fold, t)]]
        for r in generated:
            r["row_id"] = "gen:" + r["sample_id"]; r["source_group"] = CONFIG["data_source"]
        return fit + generated
    if spec["kind"] == "mixdup":
        by = {r["instance_id"]: r for r in fit}
        out = list(fit)
        for iid, n in V3["mix"]["arms"][spec["arm"]]["folds"][str(fold)]["dup"]:
            for j in range(1, n + 1):
                x = dict(by[iid]); x["row_id"] = x["row_id"] + ":dup%d" % j; x["source_group"] = "original_repeat"; out.append(x)
        return out
    if spec["kind"] == "mixgen":
        generated = [dict(r) for u in V3["mix"]["arms"][spec["arm"]]["folds"][str(fold)]["pair_uids"] for r in V3["by_uid"][u]]
        for r in generated:
            r["row_id"] = "gen:" + r["sample_id"]; r["source_group"] = CONFIG["data_source"]
        return fit + generated
    if group == "B1-Haiku":
        generated = [dict(r) for r in selected_by_fold[fold]]
        for r in generated:
            r["row_id"] = "gen:" + r["sample_id"]
            r["source_group"] = CONFIG["data_source"]
        return fit + generated
    if group == "step+1":
        duplicated = [dict(r) for r in fit]
        for r in duplicated:
            r["row_id"] = r["row_id"] + ":repeat1"
            r["source_group"] = "original_repeat"
        return fit + duplicated
    die("Unexpected group: " + str(group))


def task_path(out, group, seed, fold):
    return Path(out) / "tasks" / group.replace("+", "plus") / ("seed%d" % seed) / ("fold%d" % fold)


def envinfo():
    import torch
    import transformers
    try:
        import sklearn
        skv = sklearn.__version__
    except ImportError:
        skv = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "sklearn": skv,
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "precision": CONFIG["precision"], "platform": platform.platform()}


def pin_model(model, out, model_revision=None):
    p = Path(out) / "model_revision.json"
    if p.exists():
        frozen = jload(p)
        if frozen["model"] != model or (model_revision and frozen["revision"] != model_revision):
            die("Existing run pinned to another model / revision")
        return frozen["revision"]
    if Path(model).exists():
        revision = digest(Path(model) / "config.json")
    elif model_revision:
        revision = model_revision
    else:
        from huggingface_hub import HfApi
        try:
            revision = HfApi().model_info(model).sha
        except Exception as e:
            die("Cannot resolve immutable model commit. Enable Kaggle internet, provide --model-revision, or --model local_path. " + str(e))
    jsave(p, {"model": model, "revision": revision})
    return revision


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wf1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(4)), average="weighted", zero_division=0))


def weighted_wf1(y, pred, weights):
    """Exact frequency weighted F1 under integer family-bootstrap weights."""
    import numpy as np
    mat = np.zeros((4,4), dtype=float)
    np.add.at(mat, (y, pred), weights)
    tp = np.diag(mat)
    support = mat.sum(axis=1)
    fp = mat.sum(axis=0) - tp
    fn = support - tp
    denom = 2*tp + fp + fn
    f = np.divide(2*tp, denom, out=np.zeros(4), where=denom!=0)
    return float((f*support).sum() / support.sum())


def datasets_for(rows, tokenizer, reverse=False):
    import torch
    from torch.utils.data import Dataset
    from transformers import DataCollatorWithPadding
    class Encoded(Dataset):
        def __init__(self, seq):
            self.rows = seq
            left = [r["text2"] if reverse else r["text1"] for r in seq]
            right = [r["text1"] if reverse else r["text2"] for r in seq]
            self.ids = [L2I[r["label"]] for r in seq]
            self.enc = tokenizer(left, right, truncation=True,
                                 max_length=CONFIG["max_length"], padding=False)
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {**{k: v[i] for k,v in self.enc.items()}, "labels": self.ids[i]}
    return Encoded(rows), DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


def prediction(model, dataset, collator, batch_size=64):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    log = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator):
            y = batch.pop("labels")
            batch = {k: v.to("cuda") for k,v in batch.items()}
            # R1: frozen C0 inference is strictly FP32, for validation selection AND OOF.
            # Keep fp16 autocast ONLY for the training forward pass.
            output = model(**batch)
            l = output.logits.float().cpu().numpy()
            if not np.isfinite(l).all():
                die("Non-finite validation/test logits")
            log.append(l)
    return np.concatenate(log, axis=0)


def learning_rate_trace(total_updates):
    return {"total_scheduled_steps": total_updates,
            "warmup_updates": int(total_updates * CONFIG["warmup_ratio"]),
            "scheduler": "linear_decay_after_warmup"}


def apply_update(loss, opt, sched, scaler, params):
    """C0-v3-equivalent optimizer transition: updated / skipped / norm_overflow.

    GradScaler detects non-finite *elements*, but cannot catch a float32 global
    norm that overflows even though every element is finite. In that case we
    deliberately do NOT call optimizer.step or scheduler.step. This function
    does not retry or silently change the frozen training recipe.
    """
    import torch
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gradients = [p.grad for p in params if p.grad is not None]
    elems_finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()) if gradients else True
    if not elems_finite:
        # C0: once unscaled, GradScaler.step skips the optimizer if found_inf.
        scaler.step(opt)
        scaler.update()
        return "skipped", float("nan")
    grad_norm = torch.nn.utils.clip_grad_norm_(params, CONFIG["gradient_clip"])
    if not bool(torch.isfinite(grad_norm)):
        opt.zero_grad(set_to_none=True)
        return "norm_overflow", float(grad_norm)
    scaler.step(opt)
    scaler.update()
    sched.step()
    return "updated", float(grad_norm)


def plan_for_fold(fold, original, selected, splits):
    """Hard gate: B1-Haiku and step+1 have identical planned training schedules."""
    n_original = len(splits["folds"][str(fold)]["fit"])
    n_generated = len(selected[fold])
    b1, step1 = n_original + n_generated, 2 * n_original
    batch = CONFIG["batch_size"]
    b1_per_epoch, step_per_epoch = math.ceil(b1 / batch), math.ceil(step1 / batch)
    if b1 != step1 or b1_per_epoch != step_per_epoch:
        die("B1-Haiku / step+1 row count or planned optimizer steps DIFFER in fold %d" % fold)
    total = b1_per_epoch * CONFIG["max_epochs"]
    return {"fold": fold, "B1_rows": b1, "step1_rows": step1,
            "per_epoch_planned_steps": b1_per_epoch, "total_planned_steps": total,
            "warmup_planned_steps": int(total * CONFIG["warmup_ratio"]),
            "validation_frequency": "each completed epoch", "status": "pass"}


def stable_env_identity(info):
    """Only GPU model and precision are hard environment requirements (R3)."""
    return {"gpu": info["gpu"], "precision": info["precision"]}


def record_environment(out, info, model_revision):
    """Capture full per-run versions; don't reject a session for kernel/library drift."""
    target = Path(out) / "environment.json"
    current = {"stable": stable_env_identity(info), "model_revision": model_revision}
    if target.exists():
        prev = jload(target)
        prior_stable = prev.get("stable", {k: prev.get(k) for k in ("gpu", "precision")})
        if prior_stable != current["stable"]:
            die("GPU model/precision identity changed; refuse to mix results")
        if prev.get("model_revision", model_revision) != model_revision:
            die("Model revision changed; refuse to mix results")
    jsave(target, current)
    history_file = Path(out) / "environment_history.json"
    history = jload(history_file) if history_file.exists() else []
    details = {"environment": info, "model_revision": model_revision, "code_sha256": digest(Path(__file__))}
    if details not in history:
        history.append(details)
        jsave(history_file, history)


def one_task(group, seed, fold, original, selected, splits, fam_of, out, tokenizer, model, revision, smoke=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        die("Kaggle CUDA GPU required; no CPU training fallback")
    dest = task_path(out, group, seed, fold) if not smoke else Path(out) / "smoke" / group.replace("+", "plus")
    # R4: SMOKE is disposable; re-running after code changes must always work.
    if smoke and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    train = make_rows(group, fold, original, selected, splits)
    assert len({r["row_id"] for r in train}) == len(train)
    validate = [original[s] for s in splits["folds"][str(fold)]["val"]]
    test = [original[s] for s in splits["folds"][str(fold)]["test"]]
    train_ds, collator = datasets_for(train, tokenizer)
    val_ds, _ = datasets_for(validate, tokenizer)
    num_epochs = 1 if smoke else CONFIG["max_epochs"]
    if not smoke and group in ("B1-Haiku", "step+1"):
        plan = plan_for_fold(fold, original, selected, splits)
        if len(train) != plan["B1_rows"]:
            die("Planned comparison schedule was altered")
    for a, b in PAIRED:  # v3: paired groups must have identical rows and therefore identical planned schedules
        if group in (a, b) and not smoke:
            other = b if group == a else a
            if GROUP_SPEC[other]["kind"] in ("gen", "mixgen") and "by_uid" not in V3:
                continue  # commit 1 has no scale data; the generated side checks this pair in commit 2
            if len(make_rows(other, fold, original, selected, splits)) != len(train):
                die("Paired groups %s / %s have different training rows in fold %d" % (a, b, fold))
    spec = GROUP_SPEC[group]
    loss_kind = spec.get("loss", "ce")
    aux_lambda = float(spec.get("bibce_lambda", 0.0))  # v5: 0 for every non-BiBCE group (no extra term)
    arm = spec.get("head") or spec.get("init") or spec.get("aux")
    lam = float(spec.get("lambda_O", 1.0))
    p_O = sum(1 for r in train if r["label"] == "NEGATIVE_OTHER") / len(train)
    Z = 1.0 + (lam - 1.0) * p_O
    class_w = [1.0 / Z, 1.0 / Z, 1.0 / Z, lam / Z]
    total_steps = math.ceil(len(train)/CONFIG["batch_size"]) * num_epochs
    cfg = {"group": group, "fold": fold, "seed": seed, "model": model,
           "revision": revision, "train_rows": len(train), "val_rows": len(validate), "test_rows": len(test),
           "steps": total_steps, "epochs": num_epochs, "smoke": smoke, "config": CONFIG,
           "data_sha256": {n: digest(Path(out)/"input_data"/n) for n in ("manifest.csv", "fold_selection.csv", "split_manifest.json")},
           "official_data_sha256": digest(Path(out)/"official_reference.sha256") if (Path(out)/"official_reference.sha256").exists() else None,
           "code_sha256": digest(Path(__file__)),
           "stable_environment": stable_env_identity(envinfo()),
           "seed_policy": CONFIG["seed_note"],
           "group_spec": GROUP_SPEC[group], "v3_data_sha256": V3.get("hashes"),
           "class_weights": ([round(w, 10) for w in class_w] if lam != 1.0 else None), "loss": loss_kind}
    current_hash = canonical_json_hash(cfg)
    done_file = dest/"done.json"
    if done_file.exists() and not smoke:
        d = jload(done_file)
        if (d.get("task_fingerprint") != current_hash
            or not (dest/"predictions.csv").exists()
            or digest(dest/"predictions.csv") != d.get("predictions_sha256")
            or not (dest/"val_predictions.csv").exists()
            or digest(dest/"val_predictions.csv") != d.get("val_predictions_sha256")):
            die("Existing done.json identity mismatch or prediction corruption: " + str(done_file))
        print("[SKIP authenticated]", group, seed, fold, flush=True)
        return
    task_json = dest/"task.json"
    if task_json.exists() and jload(task_json) != cfg:
        die("Task identity changed, refusing to mix: " + str(task_json))
    jsave(task_json, cfg)
    if smoke:
        # Stress a long-batch forward/backward and include all four NLI labels.
        # This is only engineering validation; no smoke sample enters formal OOF.
        ranked = sorted(range(len(train)), key=lambda i: (len(train[i]["text1"].split()) + len(train[i]["text2"].split()), i), reverse=True)
        chosen = []
        for lab in LABELS:
            match = next((i for i in ranked if train[i]["label"] == lab), None)
            if match is not None and match not in chosen:
                chosen.append(match)
        chosen.extend(i for i in ranked if i not in chosen)
        train_ds = torch.utils.data.Subset(train_ds, chosen[:min(2*CONFIG["batch_size"], len(train_ds))])
        total_steps = 2
    init_seed = int(seed*1000 + fold)
    set_seed(init_seed)
    net = AutoModelForSequenceClassification.from_pretrained(model, revision=None if Path(model).exists() else revision,
                                num_labels=4, id2label={i:l for i,l in enumerate(LABELS)},
                                label2id=L2I, ignore_mismatched_sizes=False).float().cuda()
    net, arm_report = prepare_arm(net, spec, init_seed)  # v5: B0 / BiBCE untouched; RNG state restored
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith("bias") or "LayerNorm.weight" in n or "layer_norm.weight" in n else decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": CONFIG["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=CONFIG["lr"])
    sched = get_linear_schedule_with_warmup(optimizer, int(total_steps*CONFIG["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    epoch_log = []
    best, best_epoch = -1.0, None
    updates, skips = 0, 0
    grad_trace = []  # v4.1: per-step status and unscaled, pre-clip global grad norm (SMOKE only)
    checkpoint = dest / "best_state.pt"
    started = time.time()
    for ep in range(num_epochs):
        net.train()
        generator = torch.Generator().manual_seed(init_seed + 10000 + ep)
        loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                            generator=generator, collate_fn=collator, num_workers=0)
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(loader):
            if smoke and step >= 2:
                break
            y = batch.pop("labels").to("cuda")
            batch = {k: v.to("cuda") for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(**batch).logits
                if loss_kind == "ce":
                    if lam == 1.0:
                        loss = F.cross_entropy(logits.float(), y)  # unchanged v2 / C0 branch
                    else:
                        wv = torch.tensor(class_w, device=logits.device, dtype=torch.float32)
                        loss = (wv[y] * F.cross_entropy(logits.float(), y, reduction="none")).mean()
            if loss_kind != "ce":
                loss = new_loss(loss_kind, spec, logits, y)  # v4: fp32, outside autocast
            aux_val = None
            if aux_lambda:
                aux = bibce_loss(logits, y)  # v5: fp32, outside autocast
                aux_val = float(aux.detach())
                loss = loss + aux_lambda * aux
            if not bool(torch.isfinite(loss)):
                die("Nonfinite training loss: " + group)
            status, grad_norm = apply_update(loss, optimizer, sched, scaler, list(net.parameters()))
            if smoke:
                grad_trace.append({"epoch": ep + 1, "batch": step, "status": status, "grad_norm_unscaled_preclip": (grad_norm if math.isfinite(grad_norm) else None),
                                   "loss": float(loss.detach()), **({"aux_bibce": aux_val} if aux_val is not None else {})})
            if status == "skipped":
                skips += 1
            elif status == "updated":
                updates += 1
            elif status == "norm_overflow":
                jsave(dest/"failure.json", {"status": "norm_overflow", "epoch": ep+1,
                                               "batch": step, "group": group, "seed": seed,
                                               "fold": fold, "grad_norm": str(grad_norm),
                                               "actual_updates": updates, "skipped": skips})
                die("Finite gradient elements but nonfinite global norm: no optimizer/scheduler update; task halted")
            else:
                die("Unexpected optimizer status: " + status)
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
        val_logits = prediction(net, val_ds, collator)
        val_pred = val_logits.argmax(axis=1)
        val_score = wf1(np.array([L2I[r["label"]] for r in validate]), val_pred)
        info = {"epoch": ep+1, "val_original_wf1": val_score,
                "train_loss": total_loss / max(1,seen), "updates_total": updates, "skips_total": skips,
                "scaler_scale": scaler.get_scale()}
        epoch_log.append(info)
        print("[EPOCH]",group,seed,fold,info,flush=True)
        if val_score > best + 1e-12:
            best, best_epoch = val_score, ep+1
            torch.save(net.state_dict(), checkpoint)
    last_epoch = None
    if True:  # v3: logits of the common final step (epoch 12) for later bias / calibration analysis (SMOKE too)
        val_rev_ds, _ = datasets_for(validate, tokenizer, reverse=True)
        test_ds_l, _ = datasets_for(test, tokenizer)
        test_rev_l, _ = datasets_for(test, tokenizer, reverse=True)
        last_epoch = {"val": prediction(net, val_ds, collator), "val_rev": prediction(net, val_rev_ds, collator),
                      "test": prediction(net, test_ds_l, collator), "test_rev": prediction(net, test_rev_l, collator)}
    if not checkpoint.exists():
        die("No best checkpoint")
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    checkpoint.unlink()
    if smoke and updates < 1:
        die("SMOKE produced zero actual optimizer updates")
    if smoke and (loss_kind != "ce" or arm) and updates != 2:
        die("v4 SMOKE for %s needs exactly 2 actual optimizer updates, got %d (skips %d)" % (group, updates, skips))
    # v3: SMOKE now also runs the whole output path (test inference, CSVs, done.json) inside its own smoke directory
    test_ds, _ = datasets_for(test, tokenizer)
    reversed_ds, _ = datasets_for(test, tokenizer, reverse=True)
    plain = prediction(net, test_ds, collator)
    flip = prediction(net, reversed_ds, collator)
    def softmax(a):
        a = a.astype(float)
        exp = np.exp(a-a.max(axis=1,keepdims=True))
        return exp/exp.sum(axis=1,keepdims=True)
    pp = softmax(plain)
    rp = softmax(flip)
    sym = (pp + rp[:,SWAP]) / 2.0
    # S1: save best-epoch ORIGINAL-direction inner validation probabilities and
    # logits after reloading exactly the checkpoint used for OOF prediction.
    val_logits_best = prediction(net, val_ds, collator)
    val_probs = softmax(val_logits_best)
    best_val_again = wf1(np.array([L2I[r["label"]] for r in validate]), val_probs.argmax(axis=1))
    if abs(best_val_again - best) > 1e-12:
        die("Restored best checkpoint changed the validation selection score")
    val_output = []
    names = ("E", "F", "B", "O")
    for i, r in enumerate(validate):
        val_output.append({"instance_id": r["instance_id"], "pair_id": r["pair_id"],
                           "label": r["label"], "pred": LABELS[int(val_probs[i].argmax())],
                           **{"p_"+name:format(float(val_probs[i,j]),".10g") for j,name in enumerate(names)},
                           **{"logit_"+name:format(float(val_logits_best[i,j]),".10g") for j,name in enumerate(names)}})
    val_columns = ["instance_id", "pair_id", "label", "pred"] + ["p_"+x for x in names] + ["logit_"+x for x in names]
    csvsave(dest/"val_predictions.csv", val_output, val_columns)
    # v3: best-epoch reverse-direction inner validation logits, and epoch-12 logits for val / test (both directions)
    val_rev_best = prediction(net, datasets_for(validate, tokenizer, reverse=True)[0], collator)
    extra = []
    for part, rows_, arrs in (("val", validate, {"best_rev": val_rev_best, "last": last_epoch["val"], "last_rev": last_epoch["val_rev"]}),
                              ("test", test, {"last": last_epoch["test"], "last_rev": last_epoch["test_rev"]})):
        for i, r in enumerate(rows_):
            row = {"part": part, "instance_id": r["instance_id"]}
            for tag in ("best_rev", "last", "last_rev"):
                for j, name in enumerate(names):
                    row[tag + "_" + name] = format(float(arrs[tag][i, j]), ".10g") if tag in arrs else ""
            extra.append(row)
    csvsave(dest/"extra_logits.csv", extra, ["part", "instance_id"] + [t + "_" + n for t in ("best_rev", "last", "last_rev") for n in names])
    preds = []
    for i,r in enumerate(test):
        preds.append({"instance_id":r["instance_id"], "pair_id":r["pair_id"],
                      "reverse_pair_id":r.get("reverse_pair_id", ""),
                      "family_id":fam_of[r["pair_id"]], "label":r["label"],
                      "plain":LABELS[int(pp[i].argmax())], "sym":LABELS[int(sym[i].argmax())],
                      **{"p_"+name:format(float(pp[i,j]),".10g") for j,name in enumerate(("E","F","B","O"))},
                      **{"r_p_"+name:format(float(rp[i,j]),".10g") for j,name in enumerate(("E","F","B","O"))},
                      **{"sym_p_"+name:format(float(sym[i,j]),".10g") for j,name in enumerate(("E","F","B","O"))}})
    cols = ["instance_id","pair_id","reverse_pair_id","family_id","label","plain","sym"] + ["p_"+x for x in ("E","F","B","O")] + ["r_p_"+x for x in ("E","F","B","O")] + ["sym_p_"+x for x in ("E","F","B","O")]
    csvsave(dest/"predictions.csv", preds, cols)
    ytrue=np.array([L2I[r["label"]] for r in test])
    done = {"status":"pass", "task_fingerprint":current_hash,"group":group,"seed":seed,"fold":fold,
            "best_epoch":best_epoch,"best_val_original_wf1":best,
            "test_original_wf1":wf1(ytrue,pp.argmax(axis=1)),
            "test_sym_wf1":wf1(ytrue,sym.argmax(axis=1)),
            "epochs":epoch_log,"planned_updates":total_steps,"actual_optimizer_updates":updates,
            "skipped_fp16_updates":skips,"train_rows":len(train),"val_rows":len(validate),"test_rows":len(test),
            "lr_schedule": learning_rate_trace(total_steps),
            "source_count":dict(collections.Counter(x["source_group"] for x in train)),
            "repeated_original_row_ids":([x["instance_id"] for x in train if x["source_group"]=="original_repeat"] if group=="step+1" else []),
            "repeat_counts":dict(collections.Counter(x["instance_id"] for x in train if x["source_group"]=="original_repeat")),
            "extra_logits_sha256":digest(dest/"extra_logits.csv"), "class_weights": cfg["class_weights"],
            "predictions_sha256":digest(dest/"predictions.csv"),
            "val_predictions_sha256":digest(dest/"val_predictions.csv"),
            "seed_policy": CONFIG["seed_note"], "inference_precision": "fp32_no_autocast",
            "runtime_sec":round(time.time()-started,3),"environment":envinfo()}
    if arm_report is not None:
        done["arm_report"] = arm_report
    if spec.get("head") == "olat":
        done["olat_components"] = {"val": olat_components(net, val_ds, collator, validate),
                                   "test": olat_components(net, test_ds, collator, test)}
    jsave(done_file, done)
    print("[DONE]" if not smoke else "[SMOKE OUTPUT PATH OK]",group,seed,fold,"val=",round(best,5),"test_sym=",round(done["test_sym_wf1"],5),flush=True)
    if smoke:
        jsave(dest/"smoke.json", {"group":group,"status":"pass","updates":updates,"skips":skips,"grad_trace":grad_trace,
                                   "code_sha256": digest(Path(__file__)), "revision": revision,
                                   "stable_environment": stable_env_identity(envinfo()),
                                   "input_hashes": cfg["data_sha256"], "v3_data_sha256": V3.get("hashes"),
                                   "output_path_checked": True, "arm_report": arm_report,
                                   "validation_wf1":best,"epochs":epoch_log,"elapsed_sec":round(time.time()-started,3)})
    del net
    torch.cuda.empty_cache()


def official_score(official, rows, pred_col, target, name):
    ref = target / ("reference_"+name+".csv")
    pre = target / ("pred_"+name+".csv")
    dst = target / ("scores_"+name)
    csvsave(ref, rows, ["instance_id","pair_id","reverse_pair_id","label"])
    csvsave(pre, [{"instance_id":r["instance_id"],"label":r[pred_col]} for r in rows], ["instance_id","label"])
    if not (official / "evaluation_functions").exists():
        die("Official scoring module evaluation_functions missing; cannot finalize results")
    res = subprocess.run([sys.executable,"-m","evaluation_functions","--gold",str(ref.resolve()),"--predictions",str(pre.resolve()),"--output-dir",str(dst.resolve())],
                         cwd=official, capture_output=True, text=True)
    if res.returncode:
        die("Official scoring failed: " + res.stderr[-1500:])
    return jload(dst/"scores.json")


def finish(out, official, original, fam_of, bootstraps=2000):
    import numpy as np
    data = {}
    for seed in SEEDS:
        data[seed] = {}
        for group in GROUPS:
            frames = []
            tasks = []
            for fold in FOLDS:
                task = task_path(out,group,seed,fold)
                if not (task/"done.json").exists():
                    print("[PENDING]",group,seed,fold)
                    return {"status":"incomplete"}
                d = jload(task/"done.json")
                task_cfg = jload(task/"task.json")
                if (d.get("task_fingerprint") != canonical_json_hash(task_cfg)
                    or task_cfg.get("code_sha256") != digest(Path(__file__))):
                    die("Task code/fingerprint differs from current program: "+str(task))
                if (task_cfg.get("revision") != jload(Path(out)/"model_revision.json")["revision"]
                    or task_cfg.get("stable_environment") != jload(Path(out)/"environment.json")["stable"]):
                    die("Task model/GPU/precision differs from frozen run: "+str(task))
                for name, original_hash in task_cfg["data_sha256"].items():
                    if digest(Path(out)/"input_data"/name) != original_hash:
                        die("Task input data differs: "+name)
                if digest(task/"predictions.csv") != d["predictions_sha256"]:
                    die("Corrupt predictions: "+str(task))
                if (not (task/"val_predictions.csv").exists()
                    or digest(task/"val_predictions.csv") != d.get("val_predictions_sha256")):
                    die("Corrupt/missing validation probabilities: "+str(task))
                frames += csvrows(task/"predictions.csv")
                tasks.append(d)
            if len(frames)!=len(original) or len({x["instance_id"] for x in frames})!=len(original):
                die("Fold aggregation is not a unique, complete 3042-row OOF: "+group)
            frames.sort(key=lambda r: r["instance_id"])
            data[seed][group] = {"rows":frames,"tasks":tasks}
    first=data[SEEDS[0]][GROUPS[0]]["rows"]
    ids=[r["instance_id"] for r in first]
    if any([r["instance_id"] for r in data[s][g]["rows"]]!=ids for s in SEEDS for g in GROUPS):
        die("OOF instance ids differ across groups/seeds")
    families=sorted(set(r["family_id"] for r in first))
    fam_index={f:i for i,f in enumerate(families)}
    fam_vec=np.array([fam_index[r["family_id"]] for r in first],dtype=int)
    true=np.array([L2I[r["label"]] for r in first],dtype=int)
    if len(families)!=1725:
        print("[NOTE] Expected 1725 test families but got",len(families),flush=True)
    summaries={}
    preds={}
    for seed in SEEDS:
        summaries[str(seed)]={}
        preds[seed]={}
        for group in GROUPS:
            rows=data[seed][group]["rows"]
            pred_plain=np.array([L2I[r["plain"]] for r in rows]); pred_sym=np.array([L2I[r["sym"]] for r in rows])
            sc={"weighted_f1_plain":wf1(true,pred_plain),"weighted_f1_sym":wf1(true,pred_sym)}
            sub=Path(out)/"reports"/("seed%d"%seed)/group.replace("+","plus")
            sub.mkdir(parents=True, exist_ok=True)
            sc["official_plain"]=official_score(official,rows,"plain",sub,"plain")
            sc["official_sym"]=official_score(official,rows,"sym",sub,"sym")
            summaries[str(seed)][group]=sc
            preds[seed][group]=pred_sym
    # Cluster bootstrap: all three seeds share every resample, no pseudo-replication.
    rng=np.random.default_rng(20261106)
    differences={name:[] for name in ("B1_minus_B0","B1_minus_step1")}
    for i in range(bootstraps):
        take=rng.integers(0,len(families),len(families))
        frequency=np.bincount(take,minlength=len(families))
        w=frequency[fam_vec]
        for cmp, reference in (("B1_minus_B0","B0"),("B1_minus_step1","step+1")):
            delta=np.mean([weighted_wf1(true,preds[s]["B1-Haiku"],w) - weighted_wf1(true,preds[s][reference],w) for s in SEEDS])
            differences[cmp].append(delta)
    comp={}
    for cmp,reference in (("B1_minus_B0","B0"),("B1_minus_step1","step+1")):
        per_seed=[float(summaries[str(s)]["B1-Haiku"]["weighted_f1_sym"] - summaries[str(s)][reference]["weighted_f1_sym"]) for s in SEEDS]
        ci=np.quantile(differences[cmp],[0.025,0.975]).tolist()
        comp[cmp]={"by_seed":dict(zip(map(str,SEEDS),per_seed)),"mean_difference":float(np.mean(per_seed)),
                   "family_bootstrap_95pct_interval":ci,
                   "improvement_criterion_met":bool(np.mean(per_seed)>0 and ci[0]>0)}
    paired_updates=[]
    for seed in SEEDS:
        for fold in FOLDS:
            a=jload(task_path(out,"B1-Haiku",seed,fold)/"done.json")
            b=jload(task_path(out,"step+1",seed,fold)/"done.json")
            scheduled_equal = (a["planned_updates"] == b["planned_updates"] and
                               a["lr_schedule"] == b["lr_schedule"] and
                               a["train_rows"] == b["train_rows"] and
                               a["val_rows"] == b["val_rows"])
            for task in (a, b):
                if (len(task["epochs"]) != CONFIG["max_epochs"]
                    or [e["epoch"] for e in task["epochs"]] != list(range(1, CONFIG["max_epochs"]+1))
                    or task["actual_optimizer_updates"] + task["skipped_fp16_updates"] != task["planned_updates"]):
                    die("FROZEN PLAN VIOLATED: incomplete/mismatched epoch validation or optimizer step accounting")
            if not scheduled_equal:
                die("FROZEN PLAN VIOLATED: B1/step+1 scheduled steps, warmup, epoch validation, or rows differ")
            paired_updates.append({"seed":seed,"fold":fold,"scheduled_equal":scheduled_equal,
                                   "actual_equal":a["actual_optimizer_updates"]==b["actual_optimizer_updates"],
                                   "B1_actual":a["actual_optimizer_updates"],"step1_actual":b["actual_optimizer_updates"],
                                   "B1_skipped":a["skipped_fp16_updates"],"step1_skipped":b["skipped_fp16_updates"],
                                   "planned_updates":a["planned_updates"]})
    # S3 / R3: software versions are annotations, not hard cross-session guards.
    observed_environments = {}
    for seed in SEEDS:
        for group in GROUPS:
            for t in data[seed][group]["tasks"]:
                snapshot = t.get("environment", {})
                key = canonical_json_hash(snapshot)
                observed_environments[key] = snapshot
    environment_variations = {key: sorted({str(x.get(key)) for x in observed_environments.values()})
                              for key in ("platform", "python", "torch", "transformers", "sklearn", "cuda", "gpu", "precision")}
    result={"status":"complete","protocol":"exploratory_not_confirmatory",
            "seed_by_group":summaries,"paired_comparison":comp,
            "both_comparisons_pass":all(x["improvement_criterion_met"] for x in comp.values()),
            "test_family_count":len(families),"outer_test_sample_count":len(first),
            "seed_handling":"three seeds paired; family resampled once per bootstrap replicate and reused across groups and seeds",
            "bootstrap_replicates":bootstraps,"paired_optimizer_updates":paired_updates,
            "planned_updates_matched_all":all(p["scheduled_equal"] for p in paired_updates),
            "actual_updates_matched_all_descriptive":all(p["actual_equal"] for p in paired_updates),
            "environment_variations": environment_variations,
            "environment_versions_differ":any(len(v)>1 for v in environment_variations.values()),
            "seed_policy_note": CONFIG["seed_note"],
            "baseline_note": "B0 is a new within-experiment baseline, NOT an exact stochastic reproduction of archived C0 (different seed/shuffle stream).",
            "caveat":"After-method-selection intervals are exploratory. SMOKE is separate and never joins OOF."}
    jsave(Path(out)/"summary.json",result)
    print("[SUMMARY]",json.dumps({"both_pass":result["both_comparisons_pass"],"comparisons":comp,
                                     "planned_updates_matched_all":result["planned_updates_matched_all"],
                                     "actual_updates_matched_all_descriptive":result["actual_updates_matched_all_descriptive"]},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["preflight","smoke","full","summary"],default="preflight")
    parser.add_argument("--bundle",required=True,help="handoff folder (with data/) or original ZIP")
    parser.add_argument("--official",help="official DiCo-NLI git repo (containing final_data/)")
    parser.add_argument("--out",default="./augb_train_runs")
    parser.add_argument("--model",default=MODEL_ID)
    parser.add_argument("--model-revision",default=None)
    parser.add_argument("--fold",type=int,choices=list(FOLDS),help="run only this fold")
    parser.add_argument("--seed",type=int,choices=list(SEEDS),help="run only this seed")
    parser.add_argument("--group",choices=list(GROUPS)+["all"],default="all")
    parser.add_argument("--bootstraps",type=int,default=2000)
    parser.add_argument("--commit",choices=["1","2","3","4"],help="v3: which queue (1 = no new generated data, 2 = +2/+4 and G arms)")
    parser.add_argument("--worker",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if args.stage in ("smoke","full") and not args.commit:
        die("--commit 1 or 2 is required in v3")
    bundle=locate_bundle(args.bundle)
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    gen, selected, splits, r = preflight_generated(bundle,out)
    comparison_plans = [plan_for_fold(f, None, selected, splits) for f in FOLDS]
    jsave(out/"paired_plan_check.json", {"status": "pass", "folds": comparison_plans})
    print("[PREFLIGHT GENERATED PASS]",r["generated_pairs"],"pairs",r["generated_rows"],"rows; 5 folds checked",flush=True)
    if args.stage=="preflight" and not args.official:
        print("[PREFLIGHT PARTIAL] Original official dataset not provided, original split-family and label checks deferred.")
        return
    if not args.official:
        die("--official is required for smoke/full/summary; add official repo as Kaggle input")
    root=official_root(args.official)
    original,fam_of,ro=preflight_official(bundle,root,selected,splits,out)
    print("[PREFLIGHT OFFICIAL PASS]",ro["original_rows"],"original rows; no family leakage",flush=True)
    load_v3_data(bundle, args.commit or "1", splits)
    print("[PREFLIGHT V3 PASS] commit", args.commit or "1", "data", {k: v[:12] for k, v in V3["hashes"].items()}, flush=True)
    if args.commit in ("3", "4"):
        if not (bundle / "data_v4" / "c1_smoke_B0" / "predictions.csv").exists():
            die("Commit %s needs data_v4/c1_smoke_B0 (archived Commit 1 B0 SMOKE) for the B0 reuse check" % args.commit)
        try:
            import torch  # noqa: F401
            if args.commit == "3":
                print("[PREFLIGHT V4 LOSS CHECKS PASS] cpu", json.dumps(loss_checks("cpu")), flush=True)
            else:
                print("[PREFLIGHT V5 ARM CHECKS PASS] cpu", json.dumps(arm_checks("cpu")), flush=True)
        except ImportError:
            print("[PREFLIGHT] torch not importable here; numeric checks run again on GPU in SMOKE", flush=True)
    if args.stage=="preflight":
        return
    # Bind formal result identity to exact input files. Copy rather than referencing /kaggle/input.
    indata=out/"input_data";indata.mkdir(parents=True,exist_ok=True)
    for file in ("manifest.csv","fold_selection.csv","split_manifest.json"):
        target=indata/file
        source=bundle/"data"/file
        if target.exists() and digest(target)!=digest(source):
            die("Existing formal run uses another input: " + file)
        if not target.exists():shutil.copy2(source,target)
    shafile=out/"official_reference.sha256"
    reference_hash=digest(root/"final_data/train/dico_nli_train_track1_reference.csv")
    if shafile.exists() and shafile.read_text().strip()!=reference_hash:
        die("Official dataset changed under same result folder")
    shafile.write_text(reference_hash)
    if args.stage=="summary":
        die("v3 does not summarise in the training script; results are rebuilt offline by code/claude_recheck_augb.py")
    import torch
    if not torch.cuda.is_available():
        die("No GPU. Upload/run on Kaggle with GPU T4 selected")
    if "T4" not in torch.cuda.get_device_name(0).upper():
        die("Frozen AugB training environment requires Kaggle Tesla T4; got " + torch.cuda.get_device_name(0))
    rev=pin_model(args.model,out,args.model_revision)
    info=envinfo()
    if args.workers == 1 or args.stage == "smoke":
        record_environment(out, info, rev)
    else:  # parallel workers: read-only identity check (smoke already wrote environment.json)
        prev = jload(out/"environment.json")
        if prev["stable"] != stable_env_identity(info) or prev["model_revision"] != rev:
            die("Worker environment differs from the SMOKE environment")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=None if Path(args.model).exists() else rev)
    if args.stage=="smoke":
        import torch
        import torch.nn.functional as F
        # v3 weighted-loss sanity: lambda formula != F.cross_entropy(weight=) on a batch with O; lambda=1 equals plain CE.
        g = torch.Generator().manual_seed(0)
        lg = torch.randn(32, 4, generator=g).cuda(); yy = torch.randint(0, 4, (32,), generator=g).cuda(); yy[0] = 3
        w = torch.tensor([1/1.16, 1/1.16, 1/1.16, 2/1.16]).cuda()
        mine = (w[yy] * F.cross_entropy(lg, yy, reduction="none")).mean()
        if torch.isclose(mine, F.cross_entropy(lg, yy, weight=w)) or not torch.isclose(mine, (w[yy] * F.cross_entropy(lg, yy, reduction="none")).sum() / 32):
            die("Weighted-loss formula check failed")
        if args.commit == "3":
            lc = loss_checks("cuda")
            jsave(out/"loss_checks.json", lc)
            print("[SMOKE V4 LOSS CHECKS PASS] cuda", json.dumps(lc), flush=True)
        if args.commit == "4":
            for stale in ("arm_checks.json", "olat1_check.json", "b0_reuse.json"):
                if (out/stale).exists():
                    (out/stale).unlink()
            arm_numeric = arm_checks("cuda")
            print("[SMOKE V5 ARM NUMERIC CHECKS PASS] cuda", json.dumps(arm_numeric), flush=True)
        for group in SMOKE_GROUPS[args.commit] if args.group=="all" else [args.group]:
            one_task(group,42,0,original,selected,splits,fam_of,out,tokenizer,args.model,rev,smoke=True)
        if args.commit in ("3", "4") and args.group == "all":
            br = b0_reuse_check(bundle, out)
            print("[B0 REUSE CHECK]", br["status"], br["differences"], br["compared"], flush=True)
        if args.commit == "4" and args.group == "all":
            # real-model init checks run after the SMOKE tasks so the B0 SMOKE sees the same pre-task state as in Commit 3
            ac = {"numeric": arm_numeric, "init": arm_init_checks(args.model, rev, tokenizer, original, splits),
                  "code_sha256": digest(Path(__file__)), "status": "pass"}
            jsave(out/"arm_checks.json", ac)
            print("[SMOKE V5 ARM INIT CHECKS PASS]", json.dumps(ac["init"]), flush=True)
            oc = olat1_equivalence(out)
            print("[OLAT1 = B0 CHECK]", oc["status"], oc["differences"], oc["compared"], flush=True)
            if oc["status"] != "pass":
                die("OLat1 does not reproduce B0: the OLat head path is not equivalent; formal queue blocked")
        print("[SMOKE PASS] commit", args.commit, "groups", SMOKE_GROUPS[args.commit], "completed two optimizer steps, checkpoint restore and validation")
        return
    # Formal jobs only if all three SMOKE groups were validated in this run's environment.
    for g in SMOKE_GROUPS[args.commit]:
        sp=out/"smoke"/g.replace("+","plus")/"smoke.json"
        if not sp.exists():
            die("SMOKE not passed for "+g+". Run --stage smoke --group all first")
        sr=jload(sp)
        current_hashes={n:digest(out/"input_data"/n) for n in ("manifest.csv","fold_selection.csv","split_manifest.json")}
        if (sr.get("status") != "pass" or sr.get("code_sha256") != digest(Path(__file__))
            or sr.get("revision") != rev or sr.get("input_hashes") != current_hashes
            or sr.get("stable_environment") != stable_env_identity(info)):
            die("SMOKE identity stale after code/model/data/GPU change; rerun --stage smoke")
    queue_groups = list(COMMITS[args.commit])
    if args.commit == "4":
        # v5: arm checks and OLat1 = B0 must come from this code; OLat1 = B0 is recomputed read-only from the SMOKE artifacts
        ap = out/"arm_checks.json"
        if not ap.exists() or jload(ap).get("status") != "pass" or jload(ap).get("code_sha256") != digest(Path(__file__)):
            die("arm_checks.json missing, failed or from other code: rerun --stage smoke --group all")
        if olat1_equivalence(out, write=False)["status"] != "pass":
            die("OLat1 = B0 check fails on the SMOKE artifacts: formal queue blocked")
    if args.commit in ("3", "4"):
        # v4.1: recompute from the current archive and the SMOKE artifacts verified above (read-only); never trust a stale file
        if not (out/"smoke"/"B0"/"done.json").exists():
            die("B0 SMOKE missing: run --stage smoke --group all first")
        br = b0_reuse_check(bundle, out, write=False)
        bp = out/"b0_reuse.json"
        if bp.exists() and jload(bp).get("status") != br["status"]:
            print("[B0 REUSE] note: stored b0_reuse.json disagrees with the recomputed decision; using the recomputed one", flush=True)
        if br["status"] != "pass":
            queue_groups = ["B0"] + queue_groups
            print("[B0 REUSE] failed in SMOKE: B0 is rerun under this code (15 extra tasks)", flush=True)
        else:
            print("[B0 REUSE] passed in SMOKE: Commit 1 B0 is reused; not rerun", flush=True)
    targets=[g for g in queue_groups if args.group=="all" or g==args.group]
    seeds=[args.seed] if args.seed is not None else SEEDS
    folds=[args.fold] if args.fold is not None else FOLDS
    def cost(g):
        sp = GROUP_SPEC[g]
        return {"orig": 1, "step": sp.get("k", 0) + 1, "gen": sp.get("k", 0) + 1, "mixdup": 2, "mixgen": 2}[sp["kind"]]
    queue = sorted(((g, s, k) for s in seeds for k in folds for g in targets), key=lambda t: (-cost(t[0]), GROUPS.index(t[0]), t[1], t[2]))
    load, mine = [0.0] * args.workers, []
    for t in queue:  # deterministic longest-first assignment so both GPUs finish together
        w = min(range(args.workers), key=lambda i: (load[i], i)); load[w] += cost(t[0])
        if w == args.worker: mine.append(t)
    print("[QUEUE] worker", args.worker, "of", args.workers, ":", len(mine), "tasks, relative cost", load[args.worker], flush=True)
    for group, seed, fold in mine:
        one_task(group,seed,fold,original,selected,splits,fam_of,out,tokenizer,args.model,rev)
    print("[WORKER DONE]", args.worker, flush=True)


if __name__=="__main__":
    try: main()
    except Exception as err:
        import traceback
        traceback.print_exc()
        sys.exit(1)
