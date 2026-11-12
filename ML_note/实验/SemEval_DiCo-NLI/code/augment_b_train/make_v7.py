"""Build train_augb_v7.py from train_augb_v6.py (v6.1) by exact, once-only replacements (Commit 6: DirEnt, DEC / DEC-H against NLIinit).
Design: Claude_to_Codex_ResearchAlignment_v2.md as revised by Codex's review (11-12); freeze-plan change log, Commit 6 entry."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------- docstring, groups, queues, config
rep('''"""DiCo-NLI AugB v6.1 (Claude, 11-12;''',
    '''"""DiCo-NLI AugB v7 (Claude, 11-12): v6.1 + Commit 6 = DirEnt on B0 data with the NLI-checkpoint encoder. The four labels are
recombined from one shared directional entailment score s(x, y) ("x entails y"): with a = sigmoid(s(A, B)), b = sigmoid(s(B, A)),
P(E) = ab, P(F) = a(1 - b), P(B) = (1 - a)b, P(O) = (1 - a)(1 - b). Each training row supervises only its own order with
BCE(s(A, B), t), t = 1 iff the label is E or F (fp32, outside autocast), so the training rows and forward passes equal B0's.
DEC: new 1-dim head (own Generator, seed init_seed + 30000, N(0, initializer_range^2), bias 0). DEC-H: the NLI checkpoint's own
3-way head, s = z_entailment - logsumexp(z_neutral, z_contradiction) in fp32 (indices from the checkpoint's id2label).
Prediction rule (both orders, deterministic): an order entails iff s > 0 (s == 0 counts as non-entailment); E / F / B / O
follow from the two decisions, so the swapped pair is the F <-> B mirror by construction and plain = sym.
Selection uses each model's native four-class inner-validation WF1 (DEC: the two-order recombination). NLIinit (Commit 4) is
reused when its SMOKE equals the archived Commit 4 NLIinit SMOKE cell by cell (same mechanism as the B0 reuse); otherwise
NLIinit is rerun in the same queue. Other groups take exactly the v6.1 path.

v6.1 docstring follows.
DiCo-NLI AugB v6.1 (Claude, 11-12;''')

rep('''              "B2-NLIinit": {"kind": "gen", "k": 2, "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                             "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]}}''',
    '''              "B2-NLIinit": {"kind": "gen", "k": 2, "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                             "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]},
              # v7 (Commit 6): DirEnt on B0 data, NLIinit encoder; only the output structure and the loss change
              "DEC": {"kind": "orig", "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                      "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."],
                      "head": "dec", "head_seed_offset": 30000, "loss": "dec_bce"},
              "DEC-H": {"kind": "orig", "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."],
                        "head": "dec_h", "loss": "dec_bce"}}''')

rep('''           "5": ["B2-Haiku", "B2-NLIinit"]}  # v6: B2 rerun, no reuse''',
    '''           "5": ["B2-Haiku", "B2-NLIinit"],  # v6: B2 rerun, no reuse
           "6": ["DEC", "DEC-H"]}  # v7: NLIinit is added only if the NLIinit reuse check fails''')
rep('''"4": ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"], "5": ["B2-Haiku", "B2-NLIinit"]}''',
    '''"4": ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"], "5": ["B2-Haiku", "B2-NLIinit"],
                "6": ["NLIinit", "DEC", "DEC-H"]}''')
rep('''"code_version": "v6.1 (Claude): v5.2 + Commit 5 (B2-NLIinit against B2-Haiku); v6.1 = Commit 5 SMOKE gate after Codex's review",''',
    '''"code_version": "v7 (Claude): v6.1 + Commit 6 (DirEnt: DEC / DEC-H against NLIinit)",''')
rep('''"fp32 outside autocast; all arm changes inside torch.random.fork_rng",''',
    '''"fp32 outside autocast; all arm changes inside torch.random.fork_rng; "
                          "dec / dec_h (v7): one directional score s(x, y), P(E)=ab, P(F)=a(1-b), P(B)=(1-a)b, P(O)=(1-a)(1-b) with "
                          "a = sigmoid(s(A,B)), b = sigmoid(s(B,A)); loss BCE(s(A,B), 1[E or F]) fp32; dec = new Linear(H, 1); "
                          "dec_h = NLI 3-way head, s = z_ent - lse(z_neu, z_con); predict: order entails iff s > 0",''')

# ---------------------------------------------------------------- loss, numpy helpers, head, checks
rep('''        if kind == "soft_dice_macro4_batch":
            return dice_loss(lf, y, float(spec["eps"]))''',
    '''        if kind == "soft_dice_macro4_batch":
            return dice_loss(lf, y, float(spec["eps"]))
        if kind == "dec_bce":
            return dec_bce(lf, y)''')

rep('''_OLAT = []
''', '''_OLAT = []
_DEC = []
_NLI_HEAD = []  # v7: (weight, bias, id2label) of the pinned NLI checkpoint's own head, set by load_nli_weights for DEC-H


def dec_bce(s, y):
    """v7 DirEnt loss on fp32 scores s (rows x 1): BCE(s, t), t = 1 iff the label of this order is E or F."""
    import torch.nn.functional as F
    t = ((y == 0) | (y == 1)).to(s.dtype)
    return F.binary_cross_entropy_with_logits(s.reshape(-1), t)


def _logsig(x):
    import numpy as np
    return -np.logaddexp(0.0, -np.asarray(x, dtype=np.float64))


def dec_logp_np(s_ab, s_ba):
    """Four-class log-probabilities (E, F, B, O) of order (A, B) from the two directional scores, float64."""
    import numpy as np
    a, na, b, nb = _logsig(s_ab), _logsig(-np.asarray(s_ab, dtype=np.float64)), _logsig(s_ba), _logsig(-np.asarray(s_ba, dtype=np.float64))
    return np.stack([a + b, a + nb, na + b, na + nb], axis=1)


def dec_labels_np(s_ab, s_ba):
    """Label indices of order (A, B): an order entails iff its score is > 0 (exactly 0 = non-entailment)."""
    import numpy as np
    ea, eb = np.asarray(s_ab) > 0, np.asarray(s_ba) > 0
    return np.where(ea & eb, 0, np.where(ea, 1, np.where(eb, 2, 3))).astype(int)


def dec_head_class():
    if not _DEC:
        import torch
        class DecHead(torch.nn.Module):
            """v7: one directional entailment score. ent is None: Linear(H, 1); else the 3-way NLI head's entailment log-odds."""
            def __init__(self, hidden, n_out, ent=None, rest=None):
                super().__init__()
                self.proj = torch.nn.Linear(hidden, n_out)
                self.ent, self.rest = ent, list(rest or [])
            def forward(self, x):
                z = self.proj(x).float()
                if self.ent is None:
                    return z
                return z[..., self.ent:self.ent + 1] - torch.logsumexp(z[..., self.rest], dim=-1, keepdim=True)
        _DEC.append(DecHead)
    return _DEC[0]


def install_dec_head(net, spec, init_seed):
    """Replace net.classifier by a DecHead (inside prepare_arm's fork_rng)."""
    import torch
    old = net.classifier
    H, dev, dt = old.weight.shape[1], old.weight.device, old.weight.dtype
    if spec["head"] == "dec":
        with torch.random.fork_rng(devices=[]):  # the default Linear init would draw from the global CPU RNG
            head = dec_head_class()(H, 1).to(device=dev, dtype=dt)
        g = torch.Generator(device="cpu").manual_seed(int(init_seed) + int(spec["head_seed_offset"]))
        std = float(getattr(net.config, "initializer_range", 0.02))
        w = torch.randn(1, H, generator=g, dtype=torch.float32) * std
        with torch.no_grad():
            head.proj.weight.copy_(w.to(device=dev, dtype=dt))
            head.proj.bias.zero_()
        info = {"mode": "linear", "n_out": 1, "init_std": std, "init_seed": int(init_seed) + int(spec["head_seed_offset"]),
                "weight_abs_sum": float(w.abs().sum())}
    elif spec["head"] == "dec_h":
        if len(_NLI_HEAD) != 3:
            die("DEC-H needs the NLI checkpoint head (load_nli_weights must run first)")
        W, b, id2 = _NLI_HEAD
        names = {int(k): str(v).lower() for k, v in id2.items()}
        if sorted(names.values()) != ["contradiction", "entailment", "neutral"] or tuple(W.shape) != (3, H):
            die("NLI head labels / shape unexpected: %s %s" % (names, tuple(W.shape)))
        ent = [k for k, v in names.items() if v == "entailment"][0]
        rest = sorted(k for k, v in names.items() if v != "entailment")
        with torch.random.fork_rng(devices=[]):
            head = dec_head_class()(H, 3, ent, rest).to(device=dev, dtype=dt)
        with torch.no_grad():
            head.proj.weight.copy_(W.to(device=dev, dtype=dt))
            head.proj.bias.copy_(b.to(device=dev, dtype=dt))
        info = {"mode": "nli_head", "n_out": 3, "id2label": {str(k): v for k, v in sorted(names.items())}, "ent_index": ent, "rest_index": rest,
                "head_equal_to_checkpoint": bool(torch.equal(head.proj.weight.detach().cpu(), W.cpu()) and torch.equal(head.proj.bias.detach().cpu(), b.cpu()))}
    else:
        die("Unknown dec head " + str(spec["head"]))
    net.classifier = head
    return info


def dec_checks(device):
    """v7 numeric checks without the real model (CPU in preflight, GPU in SMOKE). Returns a report; dies on failure."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    rep, g = {"device": device}, torch.Generator().manual_seed(13)
    # E / F / B / O -> BCE target 1 / 1 / 0 / 0 against the hand formula
    s = torch.tensor([[0.3], [-0.2], [1.0], [-1.5]], device=device)
    y = torch.tensor([0, 1, 2, 3], device=device)
    t = [1.0, 1.0, 0.0, 0.0]
    exp = sum(-(ti * math.log(1 / (1 + math.exp(-si))) + (1 - ti) * math.log(1 - 1 / (1 + math.exp(-si)))) for si, ti in zip([0.3, -0.2, 1.0, -1.5], t)) / 4
    got = float(dec_bce(s, y))
    if abs(got - exp) > 1e-6:
        die("DEC BCE target mapping differs from the hand formula: %r vs %r" % (got, exp))
    rep["bce_targets_EFBO"] = [got, exp]
    # extreme scores: finite loss and gradient
    se = (torch.randn(32, 1, generator=g).sign() * 1e4).to(device).requires_grad_(True)
    ye = torch.randint(0, 4, (32,), generator=g).to(device)
    ve = dec_bce(se, ye); ge, = torch.autograd.grad(ve, se)
    if not (bool(torch.isfinite(ve)) and bool(torch.isfinite(ge).all())):
        die("DEC BCE not finite on extreme scores")
    rep["bce_extreme"] = float(ve.detach())
    # recombination: normalised, P(E)P(O) = P(F)P(B), exact F <-> B mirror under swapping, finite at extremes
    grid = np.array([-1e4, -30.0, -2.0, -1e-7, 0.0, 1e-7, 2.0, 30.0, 1e4])
    A, B = np.meshgrid(grid, grid)
    a, b = A.ravel(), B.ravel()
    lp, lr = dec_logp_np(a, b), dec_logp_np(b, a)
    norm = float(np.abs(np.logaddexp.reduce(lp, axis=1)).max())
    ident = float(np.abs((lp[:, 0] + lp[:, 3]) - (lp[:, 1] + lp[:, 2])).max())
    mirror = bool(np.array_equal(lr[:, [0, 2, 1, 3]], lp))
    if not (np.isfinite(lp).all() and norm < 1e-9 and ident < 1e-6 and mirror):
        die("DEC recombination check failed: norm %r identity %r mirror %r" % (norm, ident, mirror))
    rep["recombination"] = {"max_abs_log_norm": norm, "max_abs_log_identity": ident, "swap_mirror_exact": mirror, "cells": int(len(a))}
    # decision rule: ties (s == 0) are non-entailment; swapped labels mirror F <-> B
    la, lb = dec_labels_np(a, b), dec_labels_np(b, a)
    want = np.where((a > 0) & (b > 0), 0, np.where(a > 0, 1, np.where(b > 0, 2, 3)))
    mir = np.array([0, 2, 1, 3])[la]
    if not (np.array_equal(la, want) and np.array_equal(lb, mir)):
        die("DEC decision rule or its swap mirror is wrong")
    rep["decision_rule"] = {"zero_zero": int(dec_labels_np([0.0], [0.0])[0]), "pos_zero": int(dec_labels_np([1.0], [0.0])[0]),
                            "zero_pos": int(dec_labels_np([0.0], [1.0])[0]), "swap_mirror": True}
    if rep["decision_rule"] != {"zero_zero": 3, "pos_zero": 1, "zero_pos": 2, "swap_mirror": True}:
        die("DEC tie rule unexpected: %s" % rep["decision_rule"])
    # DEC-H score = entailment log-odds of the 3-way head, for every position of the entailment column
    z = (torch.randn(64, 3, generator=g) * 4).to(device)
    worst = 0.0
    for ent in range(3):
        rest = [k for k in range(3) if k != ent]
        sc = z[:, ent:ent + 1] - torch.logsumexp(z[:, rest], dim=-1, keepdim=True)
        worst = max(worst, float((torch.sigmoid(sc[:, 0]) - torch.softmax(z, dim=-1)[:, ent]).abs().max()))
    if worst > 1e-6:
        die("DEC-H score is not the entailment log-odds: %r" % worst)
    rep["dech_logodds_max_abs"] = worst
    # head install on a stand-in model: global RNG untouched, DEC init deterministic, DEC-H copies the given head exactly
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(16, 4)
            self.config = type("C", (), {"initializer_range": 0.02})()
    W, bb = torch.randn(3, 16, generator=g), torch.randn(3, generator=g)
    heads = []
    for _ in range(2):
        st = Stub().to(device)
        cpu0 = torch.get_rng_state()
        info = install_dec_head(st, {"head": "dec", "head_seed_offset": 30000}, 42000)
        if not torch.equal(cpu0, torch.get_rng_state()):
            die("DEC head install changed the CPU RNG state")
        heads.append(st.classifier.proj.weight.detach().cpu().clone())
    if not torch.equal(heads[0], heads[1]):
        die("DEC head init is not deterministic")
    _NLI_HEAD[:] = [W, bb, {0: "entailment", 1: "neutral", 2: "contradiction"}]
    st = Stub().to(device)
    info_h = install_dec_head(st, {"head": "dec_h"}, 42000)
    _NLI_HEAD[:] = []
    feats = torch.randn(8, 16, generator=g).to(device)
    zz = feats @ W.to(device).T + bb.to(device)
    diff = float((torch.sigmoid(st.classifier(feats)[:, 0]) - torch.softmax(zz, -1)[:, 0]).abs().max())
    if not (info_h["head_equal_to_checkpoint"] and diff < 1e-6):
        die("DEC-H head install does not reproduce the given 3-way head")
    rep["install"] = {"dec_deterministic": True, "dec_rng_untouched": True, "dech_copy_exact": True, "dech_prob_max_abs": diff}
    rep["status"] = "pass"
    return rep


DEC_NUMERIC_KEYS = ("bce_targets_EFBO", "bce_extreme", "recombination", "decision_rule", "dech_logodds_max_abs", "install")


def dec_init_checks(model, revision, tokenizer, original, splits):
    """v7 GPU checks with the real model (SMOKE): DEC-H at initialisation equals the NLI checkpoint's own entailment probability
    on 64 fold-0 fit rows (eval, fp32); DEC's new head gives finite scores. Built exactly as in one_task."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification
    rep = {}
    kw = dict(num_labels=4, id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I, ignore_mismatched_sizes=False)
    rev = None if Path(model).exists() else revision
    rows = [original[x] for x in splits["folds"]["0"]["fit"][:64]]
    ds, coll = datasets_for(rows, tokenizer)
    nli = GROUP_SPEC["DEC-H"]
    src = AutoModelForSequenceClassification.from_pretrained(nli["init_model"], revision=nli["init_revision"], torch_dtype=torch.float32).cuda()
    id2 = {int(k): str(v).lower() for k, v in src.config.id2label.items()}
    ent = [k for k, v in id2.items() if v == "entailment"][0]
    zs = prediction(src, ds, coll).astype(np.float64)
    p_ent = np.exp(zs[:, ent] - np.logaddexp.reduce(zs, axis=1))
    del src
    for g in ("DEC-H", "DEC"):
        set_seed(42 * 1000 + 0)
        net = AutoModelForSequenceClassification.from_pretrained(model, revision=rev, **kw).float().cuda()
        net, ar = prepare_arm(net, GROUP_SPEC[g], 42 * 1000 + 0)
        sc = prediction(net, ds, coll)[:, 0].astype(np.float64)
        if g == "DEC-H":
            d = float(np.abs(1 / (1 + np.exp(-sc)) - p_ent).max())
            rep[g] = {"max_abs_entail_prob_diff": d, "id2label": {str(k): v for k, v in sorted(id2.items())}, "ent_index": ent,
                      "global_rng_restored": ar["global_rng_restored"], "dec_init": ar["dec_init"]}
            if d > 1e-5:
                die("DEC-H at initialisation differs from the NLI checkpoint's entailment probability by %r" % d)
        else:
            rep[g] = {"finite": bool(np.isfinite(sc).all()), "score_abs_max": float(np.abs(sc).max()),
                      "global_rng_restored": ar["global_rng_restored"], "dec_init": ar["dec_init"]}
            if not rep[g]["finite"]:
                die("DEC scores at initialisation are not finite")
        del net
        torch.cuda.empty_cache()
    rep["status"] = "pass"
    return rep


def dec_zeroshot(model, revision, tokenizer, original, out):
    """v7 descriptive reference: DEC-H at initialisation (= the NLI checkpoint's entailment log-odds) on all 3,042 OOF rows,
    both orders; no training. Written to zeroshot_dech.csv."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification
    kw = dict(num_labels=4, id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I, ignore_mismatched_sizes=False)
    set_seed(42 * 1000 + 0)
    net = AutoModelForSequenceClassification.from_pretrained(model, revision=None if Path(model).exists() else revision, **kw).float().cuda()
    net, _ = prepare_arm(net, GROUP_SPEC["DEC-H"], 42 * 1000 + 0)
    rows = [original[k] for k in sorted(original)]
    ds, coll = datasets_for(rows, tokenizer)
    rds, _ = datasets_for(rows, tokenizer, reverse=True)
    a, b = prediction(net, ds, coll)[:, 0], prediction(net, rds, coll)[:, 0]
    csvsave(Path(out) / "zeroshot_dech.csv", [{"instance_id": r["instance_id"], "s_ab": repr(float(x)), "s_ba": repr(float(y))}
                                                for r, x, y in zip(rows, a, b)], ["instance_id", "s_ab", "s_ba"])
    del net
    torch.cuda.empty_cache()
    return {"rows": len(rows), "sha256": digest(Path(out) / "zeroshot_dech.csv")}


def dec_receipt(out, rev, info, numeric, init, zeroshot):
    return {"status": "pass", "code_sha256": digest(Path(__file__)), "revision": rev,
            "input_hashes": {n: digest(Path(out) / "input_data" / n) for n in INPUT_NAMES},
            "stable_environment": stable_env_identity(info), "software_environment": soft_env(info),
            "smoke_binding": smoke_binding(out, "6"), "numeric": numeric, "init": init, "zeroshot": zeroshot}


def commit6_smoke_problems(out, rev, info):
    """v7, Commit 6 formal entry: three SMOKE groups with exactly 2 actual updates and 0 skips, output path run, current V3 data;
    all three loaded the pinned NLI encoder; DEC / DEC-H heads are the intended ones and their parameters moved;
    dec_checks.json is complete and bound to this code, model, inputs, environment and SMOKE run; zeroshot_dech.csv intact."""
    probs = []
    for g in SMOKE_GROUPS["6"]:
        try:
            sm, dn = jload(Path(out) / "smoke" / g / "smoke.json"), jload(Path(out) / "smoke" / g / "done.json")
        except (OSError, ValueError):
            probs.append(g + " SMOKE files")
            continue
        tr = sm.get("grad_trace") or []
        if not (sm.get("status") == "pass" and sm.get("updates") == 2 and sm.get("skips") == 0 and len(tr) == 2
                and all(x.get("status") == "updated" for x in tr) and sm.get("output_path_checked") is True
                and dn.get("actual_optimizer_updates") == 2 and dn.get("skipped_fp16_updates") == 0):
            probs.append(g + " two actual updates")
        if sm.get("v3_data_sha256") != V3.get("hashes"):
            probs.append(g + " v3_data_sha256")
        r = sm.get("arm_report") or {}
        n, spec = r.get("nli_init") or {}, GROUP_SPEC[g]
        if not (r.get("global_rng_restored") is True and n.get("model") == spec["init_model"] and n.get("revision") == spec["init_revision"]
                and n.get("tensors_loaded") == 200 and n.get("parameters_loaded") == 278809344
                and n.get("kept_from_b0") == ["classifier.bias", "classifier.weight"] and n.get("forward_config_differences_kept_b0") == {}):
            probs.append(g + " nli_init")
        if g in ("DEC", "DEC-H"):
            di = r.get("dec_init") or {}
            want = ("linear", 1) if g == "DEC" else ("nli_head", 3)
            if (di.get("mode"), di.get("n_out")) != want or (g == "DEC-H" and di.get("head_equal_to_checkpoint") is not True):
                probs.append(g + " dec_init")
            hc = dn.get("dec_head_change") or {}
            if not (isinstance(hc, dict) and hc and all(_fin(v, 0.0) for v in hc.values()) and _fin(hc.get("proj.weight"), 1e-12)):
                probs.append(g + " head parameters did not move")
    p = Path(out) / "dec_checks.json"
    if not p.exists():
        return probs + ["dec_checks.json missing"]
    a = jload(p)
    if a.get("status") != "pass" or a.get("code_sha256") != digest(Path(__file__)) or a.get("revision") != rev:
        probs.append("dec_checks identity")
    if a.get("input_hashes") != {n: digest(Path(out) / "input_data" / n) for n in INPUT_NAMES}:
        probs.append("dec_checks input_hashes")
    if a.get("stable_environment") != stable_env_identity(info) or a.get("software_environment") != soft_env(info):
        probs.append("dec_checks environment")
    sb = smoke_binding(out, "6")
    if None in sb.values() or a.get("smoke_binding") != sb:
        probs.append("dec_checks smoke_binding")
    nm = a.get("numeric") or {}
    if nm.get("status") != "pass" or any(k not in nm for k in DEC_NUMERIC_KEYS) or nm.get("decision_rule") != {
            "zero_zero": 3, "pos_zero": 1, "zero_pos": 2, "swap_mirror": True} or not _fin(nm.get("dech_logodds_max_abs"), 0.0, 1e-6):
        probs.append("dec_checks numeric")
    it = a.get("init") or {}
    if it.get("status") != "pass" or not _fin((it.get("DEC-H") or {}).get("max_abs_entail_prob_diff"), 0.0, 1e-5) \\
            or (it.get("DEC") or {}).get("finite") is not True:
        probs.append("dec_checks init")
    z = a.get("zeroshot") or {}
    zp = Path(out) / "zeroshot_dech.csv"
    if z.get("rows") != 3042 or not zp.exists() or digest(zp) != z.get("sha256"):
        probs.append("zeroshot_dech.csv")
    return probs
''')

# ---------------------------------------------------------------- NLI head stash, install in prepare_arm
rep('''    del src
    return info
''', '''    _NLI_HEAD[:] = [src.classifier.weight.detach().cpu().clone(), src.classifier.bias.detach().cpu().clone(), dict(src.config.id2label)]  # v7 DEC-H
    del src
    return info
''')
rep('''        if spec.get("head") == "olat":
            report["olat_init"] = install_olat_head(net, spec, init_seed)''',
    '''        if spec.get("head") == "olat":
            report["olat_init"] = install_olat_head(net, spec, init_seed)
        if spec.get("head") in ("dec", "dec_h"):
            report["dec_init"] = install_dec_head(net, spec, init_seed)
    _NLI_HEAD[:] = []  # v7: the stashed NLI head is only used inside this call''')

# ---------------------------------------------------------------- NLIinit reuse = the B0 reuse mechanism with other paths
rep('''def b0_reuse_check(bundle, out, write=True):''',
    '''def b0_reuse_check(bundle, out, write=True, ref_rel="data_v4/c1_smoke_B0", group="B0", report_name="b0_reuse.json"):''')
rep('''    ref, new = Path(bundle) / "data_v4" / "c1_smoke_B0", Path(out) / "smoke" / "B0"''',
    '''    ref, new = Path(bundle) / ref_rel, Path(out) / "smoke" / group  # v7: also NLIinit against the Commit 4 archive''')
rep('''        jsave(Path(out) / "b0_reuse.json", report)''', '''        jsave(Path(out) / report_name, report)''')

# ---------------------------------------------------------------- one_task: DEC path (other groups untouched)
rep('''    val_ds, _ = datasets_for(validate, tokenizer)
''', '''    val_ds, _ = datasets_for(validate, tokenizer)
    is_dec = GROUP_SPEC[group].get("head") in ("dec", "dec_h")  # v7
    val_rev_ds0 = datasets_for(validate, tokenizer, reverse=True)[0] if is_dec else None
''')
rep('''    checkpoint = dest / "best_state.pt"
''', '''    checkpoint = dest / "best_state.pt"
    head0 = {k: v.detach().cpu().clone() for k, v in net.classifier.state_dict().items()} if is_dec else None  # v7
''')
rep('''        val_logits = prediction(net, val_ds, collator)
        val_pred = val_logits.argmax(axis=1)''',
    '''        val_logits = prediction(net, val_ds, collator)
        val_pred = val_logits.argmax(axis=1) if not is_dec else dec_labels_np(val_logits[:, 0], prediction(net, val_rev_ds0, collator)[:, 0])''')
rep('''                      "test": prediction(net, test_ds_l, collator), "test_rev": prediction(net, test_rev_l, collator)}
''', '''                      "test": prediction(net, test_ds_l, collator), "test_rev": prediction(net, test_rev_l, collator)}
        if is_dec:  # v7: store the recombined four-class log-probabilities of each order
            for k1, k2 in (("val", "val_rev"), ("test", "test_rev")):
                x1, x2 = last_epoch[k1][:, 0].copy(), last_epoch[k2][:, 0].copy()
                last_epoch[k1], last_epoch[k2] = dec_logp_np(x1, x2), dec_logp_np(x2, x1)
''')
rep('''    checkpoint.unlink()
''', '''    checkpoint.unlink()
    head_change = ({k: float((net.classifier.state_dict()[k].detach().cpu() - v).abs().max()) for k, v in head0.items()}
                   if is_dec else None)  # v7
''')
rep('''    flip = prediction(net, reversed_ds, collator)
''', '''    flip = prediction(net, reversed_ds, collator)
    if is_dec:  # v7: plain = sym by construction; labels by the s > 0 rule
        s_test, s_test_rev = plain[:, 0].astype(float).copy(), flip[:, 0].astype(float).copy()
        plain, flip = dec_logp_np(s_test, s_test_rev), dec_logp_np(s_test_rev, s_test)
''')
rep('''    val_logits_best = prediction(net, val_ds, collator)
    val_probs = softmax(val_logits_best)
    best_val_again = wf1(np.array([L2I[r["label"]] for r in validate]), val_probs.argmax(axis=1))''',
    '''    val_logits_best = prediction(net, val_ds, collator)
    if is_dec:
        s_v, s_vr = val_logits_best[:, 0].astype(float).copy(), prediction(net, val_rev_ds0, collator)[:, 0].astype(float).copy()
        val_logits_best = dec_logp_np(s_v, s_vr)
    val_probs = softmax(val_logits_best)
    val_argmax = dec_labels_np(s_v, s_vr) if is_dec else val_probs.argmax(axis=1)
    best_val_again = wf1(np.array([L2I[r["label"]] for r in validate]), val_argmax)''')
rep('''"pred": LABELS[int(val_probs[i].argmax())],''', '''"pred": LABELS[int(val_argmax[i])],''')
rep('''                           **{"logit_"+name:format(float(val_logits_best[i,j]),".10g") for j,name in enumerate(names)}})''',
    '''                           **{"logit_"+name:format(float(val_logits_best[i,j]),".10g") for j,name in enumerate(names)},
                           **({"s_ab": repr(float(s_v[i])), "s_ba": repr(float(s_vr[i]))} if is_dec else {})})''')
rep('''    val_columns = ["instance_id", "pair_id", "label", "pred"] + ["p_"+x for x in names] + ["logit_"+x for x in names]''',
    '''    val_columns = ["instance_id", "pair_id", "label", "pred"] + ["p_"+x for x in names] + ["logit_"+x for x in names] + (["s_ab", "s_ba"] if is_dec else [])''')
rep('''    val_rev_best = prediction(net, datasets_for(validate, tokenizer, reverse=True)[0], collator)
''', '''    val_rev_best = prediction(net, datasets_for(validate, tokenizer, reverse=True)[0], collator)
    if is_dec:
        val_rev_best = dec_logp_np(s_vr, s_v)
''')
rep('''    preds = []
    for i,r in enumerate(test):''', '''    plain_idx, sym_idx = (dec_labels_np(s_test, s_test_rev),) * 2 if is_dec else (pp.argmax(axis=1), sym.argmax(axis=1))  # v7
    preds = []
    for i,r in enumerate(test):''')
rep('''"plain":LABELS[int(pp[i].argmax())], "sym":LABELS[int(sym[i].argmax())],''',
    '''"plain":LABELS[int(plain_idx[i])], "sym":LABELS[int(sym_idx[i])],
                      **({"s_ab": repr(float(s_test[i])), "s_ba": repr(float(s_test_rev[i]))} if is_dec else {}),''')
rep('''    cols = ["instance_id","pair_id","reverse_pair_id","family_id","label","plain","sym"]''',
    '''    cols = ["instance_id","pair_id","reverse_pair_id","family_id","label","plain","sym"] + (["s_ab", "s_ba"] if is_dec else [])''')
rep('''"test_original_wf1":wf1(ytrue,pp.argmax(axis=1)),''', '''"test_original_wf1":wf1(ytrue,plain_idx),''')
rep('''"test_sym_wf1":wf1(ytrue,sym.argmax(axis=1)),''', '''"test_sym_wf1":wf1(ytrue,sym_idx),''')
rep('''    if arm_report is not None:
''', '''    if is_dec:
        done["dec_head_change"] = head_change
        done["dec_rule"] = "an order entails iff s > 0; plain = sym"
    if arm_report is not None:
''')
rep('''        if RUN_COMMIT and RUN_COMMIT[0] in ("4", "5") and soft_env(d.get("environment")) != soft_env(envinfo()):''',
    '''        if RUN_COMMIT and RUN_COMMIT[0] in ("4", "5", "6") and soft_env(d.get("environment")) != soft_env(envinfo()):''')

# ---------------------------------------------------------------- main: preflight, SMOKE, formal gate
rep('''    parser.add_argument("--commit",choices=["1","2","3","4","5"],''', '''    parser.add_argument("--commit",choices=["1","2","3","4","5","6"],''')
rep('''    if args.stage=="preflight":
        return
    # Bind formal result identity''', '''    if args.commit == "6":
        if not (bundle / "data_v7" / "c4_smoke_NLIinit" / "predictions.csv").exists():
            die("Commit 6 needs data_v7/c4_smoke_NLIinit (archived Commit 4 NLIinit SMOKE) for the NLIinit reuse check")
        try:
            import torch  # noqa: F401
            print("[PREFLIGHT V7 DEC CHECKS PASS] cpu", json.dumps(dec_checks("cpu")), flush=True)
        except ImportError:
            print("[PREFLIGHT] torch not importable here; DEC numeric checks run again on GPU in SMOKE", flush=True)
    if args.stage=="preflight":
        return
    # Bind formal result identity''')
rep('''        for group in SMOKE_GROUPS[args.commit] if args.group=="all" else [args.group]:''',
    '''        if args.commit == "6":
            for stale in ("dec_checks.json", "nli_reuse.json", "zeroshot_dech.csv"):
                if (out/stale).exists():
                    (out/stale).unlink()
            dec_numeric = dec_checks("cuda")
            print("[SMOKE V7 DEC NUMERIC CHECKS PASS] cuda", json.dumps(dec_numeric), flush=True)
        for group in SMOKE_GROUPS[args.commit] if args.group=="all" else [args.group]:''')
rep('''        print("[SMOKE PASS] commit", args.commit,''', '''        if args.commit == "6" and args.group == "all":
            nr = b0_reuse_check(bundle, out, ref_rel="data_v7/c4_smoke_NLIinit", group="NLIinit", report_name="nli_reuse.json")
            print("[NLIINIT REUSE CHECK]", nr["status"], nr["differences"], nr["compared"], flush=True)
            di = dec_init_checks(args.model, rev, tokenizer, original, splits)
            print("[SMOKE V7 DEC INIT CHECKS PASS]", json.dumps(di), flush=True)
            zs = dec_zeroshot(args.model, rev, tokenizer, original, out)
            print("[ZERO-SHOT DEC-H WRITTEN]", json.dumps(zs), flush=True)
            jsave(out/"dec_checks.json", dec_receipt(out, rev, info, dec_numeric, di, zs))
        print("[SMOKE PASS] commit", args.commit,''')
rep('''    if args.commit in ("4", "5"):
        # v5.1''', '''    if args.commit in ("4", "5", "6"):
        # v5.1''')
rep('''        if args.commit == "5":
            probs = commit5_smoke_problems(out)''', '''        if args.commit == "6":
            probs = commit6_smoke_problems(out, rev, info)
            if probs:
                die("Commit 6 SMOKE records do not validate %s: rerun --stage smoke --group all" % probs)
        if args.commit == "5":
            probs = commit5_smoke_problems(out)''')
rep('''        stale = env_mismatched_tasks(out, (["B0"] if args.commit == "4" else []) + COMMITS[args.commit], soft_env(info))''',
    '''        stale = env_mismatched_tasks(out, (["B0"] if args.commit == "4" else ["NLIinit"] if args.commit == "6" else []) + COMMITS[args.commit], soft_env(info))''')
rep('''    targets=[g for g in queue_groups if args.group=="all" or g==args.group]''', '''    if args.commit == "6":
        nr = b0_reuse_check(bundle, out, write=False, ref_rel="data_v7/c4_smoke_NLIinit", group="NLIinit", report_name="nli_reuse.json")
        if nr["status"] != "pass":
            queue_groups = ["NLIinit"] + queue_groups
            print("[NLIINIT REUSE] failed in SMOKE: NLIinit is rerun under this code (15 extra tasks)", flush=True)
        else:
            print("[NLIINIT REUSE] passed in SMOKE: Commit 4 NLIinit is reused; not rerun", flush=True)
    targets=[g for g in queue_groups if args.group=="all" or g==args.group]''')

# ---------------------------------------------------------------- v7: control identity (15 archived Commit 4 NLIinit tasks)
rep('''def commit6_smoke_problems(out, rev, info):''', '''NLI_CONTROL_IGNORE_TASK = ("code_sha256",)


def nli_control_problems(bundle, out, original, selected, splits, rev, info):
    """v7: the 15 archived Commit 4 NLIinit tasks (data_v7/c4_nliinit_tasks) match, seed by seed and fold by fold, what this code
    and these inputs would run for NLIinit, and DEC / DEC-H train on exactly the same rows. Empty list = reusable control."""
    probs, root = [], Path(bundle) / "data_v7" / "c4_nliinit_tasks"
    cur = {n: digest(Path(out) / "input_data" / n) for n in INPUT_NAMES}
    shaf = Path(out) / "official_reference.sha256"
    off = digest(shaf) if shaf.exists() else None
    keep = lambda c: {a: b for a, b in c.items() if a not in B0_REUSE_IGNORE_CONFIG}
    for seed in SEEDS:
        for fold in FOLDS:
            tag = "seed%d/fold%d" % (seed, fold)
            try:
                d = root / ("seed%d" % seed) / ("fold%d" % fold)
                rt, rd = jload(d / "task.json"), jload(d / "done.json")
            except (OSError, ValueError):
                probs.append(tag + " archive files")
                continue
            rows = make_rows("NLIinit", fold, original, selected, splits)
            ids = [r["row_id"] for r in rows]
            for g in ("DEC", "DEC-H"):
                if [r["row_id"] for r in make_rows(g, fold, original, selected, splits)] != ids:
                    probs.append(tag + " " + g + " rows differ from NLIinit")
            want = {"group": "NLIinit", "seed": seed, "fold": fold, "smoke": False, "epochs": CONFIG["max_epochs"], "loss": "ce",
                    "class_weights": None, "group_spec": GROUP_SPEC["NLIinit"], "model": MODEL_ID, "revision": rev,
                    "train_rows": len(rows), "val_rows": len(splits["folds"][str(fold)]["val"]),
                    "test_rows": len(splits["folds"][str(fold)]["test"]),
                    "steps": math.ceil(len(rows) / CONFIG["batch_size"]) * CONFIG["max_epochs"],
                    "data_sha256": cur, "official_data_sha256": off, "stable_environment": stable_env_identity(info),
                    "v3_data_sha256": V3.get("hashes"), "seed_policy": CONFIG["seed_note"]}
            for k, v in want.items():
                if rt.get(k) != v:
                    probs.append(tag + " task." + k)
            if keep(rt.get("config") or {}) != keep(CONFIG):
                probs.append(tag + " task.config")
            extra = sorted(set(rt) - set(want) - {"config"} - set(NLI_CONTROL_IGNORE_TASK))
            if extra:
                probs.append(tag + " task has unexpected keys %s" % extra)
            if (rd.get("status") != "pass" or rd.get("task_fingerprint") != canonical_json_hash(rt)
                    or (rd.get("group"), rd.get("seed"), rd.get("fold")) != ("NLIinit", seed, fold)
                    or rd.get("planned_updates") != want["steps"] or not rd.get("predictions_sha256")
                    or soft_env(rd.get("environment")) != soft_env(info)):
                probs.append(tag + " done")
    return probs


def commit6_smoke_problems(out, rev, info):''')

# ---------------------------------------------------------------- v7: timing (descriptive; not part of any identity or reuse check)
rep('''    epoch_log = []
''', '''    epoch_log = []
    timing = {"train_sec": 0.0, "epoch_validation_sec": 0.0}  # v7
''')
rep('''        for step, batch in enumerate(loader):
''', '''        t_ep0 = time.time()  # v7
        for step, batch in enumerate(loader):
''')
rep('''        val_logits = prediction(net, val_ds, collator)
''', '''        timing["train_sec"] += time.time() - t_ep0
        t_val0 = time.time()
        val_logits = prediction(net, val_ds, collator)
''')
rep('''        val_score = wf1(np.array([L2I[r["label"]] for r in validate]), val_pred)
''', '''        val_score = wf1(np.array([L2I[r["label"]] for r in validate]), val_pred)
        timing["epoch_validation_sec"] += time.time() - t_val0
''')
rep('''    last_epoch = None
''', '''    last_epoch = None
    t_last0 = time.time()
''')
rep('''    if not checkpoint.exists():
        die("No best checkpoint")''', '''    timing["last_epoch_logits_sec"] = time.time() - t_last0
    if not checkpoint.exists():
        die("No best checkpoint")''')
rep('''    test_ds, _ = datasets_for(test, tokenizer)
    reversed_ds, _ = datasets_for(test, tokenizer, reverse=True)
''', '''    t_fin0 = time.time()
    test_ds, _ = datasets_for(test, tokenizer)
    reversed_ds, _ = datasets_for(test, tokenizer, reverse=True)
''')
rep('''    plain_idx, sym_idx =''', '''    timing["final_prediction_sec"] = time.time() - t_fin0  # test both orders, best-epoch val both orders, val CSV
    plain_idx, sym_idx =''')
rep('''    if is_dec:
        done["dec_head_change"] = head_change''', '''    done["timing"] = {k: round(v, 3) for k, v in timing.items()}
    if is_dec:
        done["dec_head_change"] = head_change''')

# ---------------------------------------------------------------- v7: SMOKE receipt and formal reuse decision use the control identity
rep('''            jsave(out/"dec_checks.json", dec_receipt(out, rev, info, dec_numeric, di, zs))''',
    '''            cp = nli_control_problems(bundle, out, original, selected, splits, rev, info)
            print("[NLIINIT CONTROL IDENTITY, 15 TASKS]", "pass" if not cp else cp, flush=True)
            rc = dec_receipt(out, rev, info, dec_numeric, di, zs)
            rc["control_identity"] = {"status": "pass" if not cp else "fail", "problems": cp, "tasks": len(SEEDS) * len(FOLDS)}
            jsave(out/"dec_checks.json", rc)''')
rep('''        if nr["status"] != "pass":
            queue_groups = ["NLIinit"] + queue_groups
            print("[NLIINIT REUSE] failed in SMOKE: NLIinit is rerun under this code (15 extra tasks)", flush=True)''',
    '''        cp = nli_control_problems(bundle, out, original, selected, splits, rev, info)
        if nr["status"] != "pass" or cp:
            queue_groups = ["NLIinit"] + queue_groups
            print("[NLIINIT REUSE] SMOKE %s, control identity %s: NLIinit is rerun under this code (15 extra tasks)"
                  % (nr["status"], "pass" if not cp else cp[:5]), flush=True)''')
rep('''    zp = Path(out) / "zeroshot_dech.csv"''', '''    ci = a.get("control_identity") or {}
    if ci.get("status") not in ("pass", "fail") or ci.get("tasks") != len(SEEDS) * len(FOLDS):
        probs.append("dec_checks control_identity")
    zp = Path(out) / "zeroshot_dech.csv"''')

open(dst, "w", encoding="utf-8").write(s)
print("ok", dst)
