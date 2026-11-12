"""Build train_augb_v5.py from train_augb_v4.py (v4.1) by exact, once-only replacements (Commit 4: three arms on B0 data)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)


rep('"""DiCo-NLI AugB v4.1 (Claude, 11-10):',
    '"""DiCo-NLI AugB v5 (Claude, 11-12): v4.1 + three arms on B0 data for Commit 4 (OLat4, NLIinit, BiBCE).\n'
    'Every arm starts from the B0 construction (same seed, same from_pretrained call, same 4-class head init); arm changes\n'
    'run inside torch.random.fork_rng so the global CPU / CUDA RNG state after construction equals B0\'s. The CE line and\n'
    'the rest of the training path are unchanged; B0 (no arm fields) takes exactly the v4.1 path. SMOKE adds OLat1\n'
    '(K = 1, no perturbation), which must reproduce the B0 SMOKE cell by cell before the formal queue starts.\n\n'
    'v4.1 docstring follows.\nDiCo-NLI AugB v4.1 (Claude, 11-10):')

rep('''              "Dice": {"kind": "orig", "loss": "soft_dice_macro4_batch", "eps": 1e-6}}
GROUPS = tuple(GROUP_SPEC)''',
    '''              "Dice": {"kind": "orig", "loss": "soft_dice_macro4_batch", "eps": 1e-6},
              # v5 (Commit 4). OLat1 is SMOKE-only: K = 1 without perturbation must equal B0 exactly.
              "OLat4": {"kind": "orig", "head": "olat", "K": 4, "perturb_std": 0.002, "perturb_seed_offset": 20000},
              "OLat1": {"kind": "orig", "head": "olat", "K": 1, "perturb_std": 0.0, "perturb_seed_offset": 20000},
              "NLIinit": {"kind": "orig", "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                          "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]},
              "BiBCE": {"kind": "orig", "aux": "bibce_mean_of_two_directions", "bibce_lambda": 0.25}}
GROUPS = tuple(GROUP_SPEC)''')

rep('''           "3": ["Focal", "Dice"]}  # v4: B0 is added only if the B0 reuse check fails
SMOKE_GROUPS = {"1": ["B0", "W2", "step+4", "Dup-O1"], "2": ["B4-Haiku", "G-O1"], "3": ["B0", "Focal", "Dice"]}''',
    '''           "3": ["Focal", "Dice"],  # v4: B0 is added only if the B0 reuse check fails
           "4": ["OLat4", "NLIinit", "BiBCE"]}  # v5: same rule
SMOKE_GROUPS = {"1": ["B0", "W2", "step+4", "Dup-O1"], "2": ["B4-Haiku", "G-O1"], "3": ["B0", "Focal", "Dice"],
                "4": ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"]}''')

rep('''          "code_version": "v4.1 (Claude): v3 + Focal / Dice loss branches",''',
    '''          "code_version": "v5 (Claude): v4.1 + OLat / NLIinit / BiBCE arms",
          "arm_variants": "olat: classifier -> Linear(H, 3+K); z_O = logsumexp(z_O1..K) - log K in fp32; init = B0 head, O rows = B0 O row + "
                          "N(0, std^2) perturbation centred over components (own Generator, seed init_seed+20000), biases copied; "
                          "nli: after the B0 construction, deberta.* and pooler.* are overwritten from the pinned NLI checkpoint "
                          "(B0 tokenizer, 4-class head and forward config kept); bibce: L = CE + lambda * 1/2 [BCE(m_AB, t_AB) + BCE(m_BA, t_BA)], "
                          "m_AB = lse(z_E, z_F) - lse(z_B, z_O), m_BA = lse(z_E, z_B) - lse(z_F, z_O), targets E(1,1) F(1,0) B(0,1) O(0,0), "
                          "fp32 outside autocast; all arm changes inside torch.random.fork_rng",''')

rep('''B0_REUSE_IGNORE_CONFIG = ("code_version", "loss_variants")''',
    '''B0_REUSE_IGNORE_CONFIG = ("code_version", "loss_variants", "arm_variants")''')

rep('''B0_REUSE_IGNORE_TASK = ("code_sha256", "loss")''', '''def _bibce(z, y):
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


B0_REUSE_IGNORE_TASK = ("code_sha256", "loss")''')

rep('''    spec = GROUP_SPEC[group]
    loss_kind = spec.get("loss", "ce")''', '''    spec = GROUP_SPEC[group]
    loss_kind = spec.get("loss", "ce")
    aux_lambda = float(spec.get("bibce_lambda", 0.0))  # v5: 0 for every non-BiBCE group (no extra term)
    arm = spec.get("head") or spec.get("init") or spec.get("aux")''')

rep('''                                label2id=L2I, ignore_mismatched_sizes=False).float().cuda()''',
    '''                                label2id=L2I, ignore_mismatched_sizes=False).float().cuda()
    net, arm_report = prepare_arm(net, spec, init_seed)  # v5: B0 / BiBCE untouched; RNG state restored''')

rep('''            if loss_kind != "ce":
                loss = new_loss(loss_kind, spec, logits, y)  # v4: fp32, outside autocast''',
    '''            if loss_kind != "ce":
                loss = new_loss(loss_kind, spec, logits, y)  # v4: fp32, outside autocast
            aux_val = None
            if aux_lambda:
                aux = bibce_loss(logits, y)  # v5: fp32, outside autocast
                aux_val = float(aux.detach())
                loss = loss + aux_lambda * aux''')

rep('''                                   "loss": float(loss.detach())})''',
    '''                                   "loss": float(loss.detach()), **({"aux_bibce": aux_val} if aux_val is not None else {})})''')

rep('''    if smoke and loss_kind != "ce" and updates != 2:''', '''    if smoke and (loss_kind != "ce" or arm) and updates != 2:''')

rep('''    jsave(done_file, done)''', '''    if arm_report is not None:
        done["arm_report"] = arm_report
    if spec.get("head") == "olat":
        done["olat_components"] = {"val": olat_components(net, val_ds, collator, validate),
                                   "test": olat_components(net, test_ds, collator, test)}
    jsave(done_file, done)''')

rep('''                                   "output_path_checked": True,''',
    '''                                   "output_path_checked": True, "arm_report": arm_report,''')

rep('''    parser.add_argument("--commit",choices=["1","2","3"],''', '''    parser.add_argument("--commit",choices=["1","2","3","4"],''')

rep('''    if args.commit == "3":
        if not (bundle / "data_v4" / "c1_smoke_B0" / "predictions.csv").exists():
            die("Commit 3 needs data_v4/c1_smoke_B0 (archived Commit 1 B0 SMOKE) for the B0 reuse check")
        try:
            import torch  # noqa: F401
            print("[PREFLIGHT V4 LOSS CHECKS PASS] cpu", json.dumps(loss_checks("cpu")), flush=True)
        except ImportError:
            print("[PREFLIGHT V4] torch not importable here; loss checks run again on GPU in SMOKE", flush=True)''',
    '''    if args.commit in ("3", "4"):
        if not (bundle / "data_v4" / "c1_smoke_B0" / "predictions.csv").exists():
            die("Commit %s needs data_v4/c1_smoke_B0 (archived Commit 1 B0 SMOKE) for the B0 reuse check" % args.commit)
        try:
            import torch  # noqa: F401
            if args.commit == "3":
                print("[PREFLIGHT V4 LOSS CHECKS PASS] cpu", json.dumps(loss_checks("cpu")), flush=True)
            else:
                print("[PREFLIGHT V5 ARM CHECKS PASS] cpu", json.dumps(arm_checks("cpu")), flush=True)
        except ImportError:
            print("[PREFLIGHT] torch not importable here; numeric checks run again on GPU in SMOKE", flush=True)''')

rep('''        if args.commit == "3":
            lc = loss_checks("cuda")
            jsave(out/"loss_checks.json", lc)
            print("[SMOKE V4 LOSS CHECKS PASS] cuda", json.dumps(lc), flush=True)''',
    '''        if args.commit == "3":
            lc = loss_checks("cuda")
            jsave(out/"loss_checks.json", lc)
            print("[SMOKE V4 LOSS CHECKS PASS] cuda", json.dumps(lc), flush=True)
        if args.commit == "4":
            for stale in ("arm_checks.json", "olat1_check.json", "b0_reuse.json"):
                if (out/stale).exists():
                    (out/stale).unlink()
            arm_numeric = arm_checks("cuda")
            print("[SMOKE V5 ARM NUMERIC CHECKS PASS] cuda", json.dumps(arm_numeric), flush=True)''')

rep('''        if args.commit == "3" and args.group == "all":
            br = b0_reuse_check(bundle, out)
            print("[B0 REUSE CHECK]", br["status"], br["differences"], br["compared"], flush=True)''',
    '''        if args.commit in ("3", "4") and args.group == "all":
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
                die("OLat1 does not reproduce B0: the OLat head path is not equivalent; formal queue blocked")''')

rep('''    queue_groups = list(COMMITS[args.commit])
    if args.commit == "3":''', '''    queue_groups = list(COMMITS[args.commit])
    if args.commit == "4":
        # v5: arm checks and OLat1 = B0 must come from this code; OLat1 = B0 is recomputed read-only from the SMOKE artifacts
        ap = out/"arm_checks.json"
        if not ap.exists() or jload(ap).get("status") != "pass" or jload(ap).get("code_sha256") != digest(Path(__file__)):
            die("arm_checks.json missing, failed or from other code: rerun --stage smoke --group all")
        if olat1_equivalence(out, write=False)["status"] != "pass":
            die("OLat1 = B0 check fails on the SMOKE artifacts: formal queue blocked")
    if args.commit in ("3", "4"):''')

rep('''            print("[B0 REUSE] failed in SMOKE: B0 is rerun under v4 (15 extra tasks)", flush=True)''',
    '''            print("[B0 REUSE] failed in SMOKE: B0 is rerun under this code (15 extra tasks)", flush=True)''')

# ---------- v5.1 (Codex review 11-12): software identity of workers / resumed tasks, receipt integrity and binding ----------
rep('''"""DiCo-NLI AugB v5 (Claude, 11-12): v4.1 + three arms on B0 data for Commit 4 (OLat4, NLIinit, BiBCE).''',
    '''"""DiCo-NLI AugB v5.1 (Claude, 11-12): v4.1 + three arms on B0 data for Commit 4 (OLat4, NLIinit, BiBCE).
v5.1 (after Codex's review): for Commit 4 the formal queue also requires torch / cuda / transformers of the worker to equal
the SMOKE run's (else rerun SMOKE in this environment first), refuses restored done.json trained under other versions, and
accepts arm_checks.json only when both sub-receipts are complete and it is bound to this SMOKE run (code, model revision,
input hashes, environment and the digests of this run's SMOKE done.json files).''')

rep('''          "code_version": "v5 (Claude): v4.1 + OLat / NLIinit / BiBCE arms",''',
    '''          "code_version": "v5.1 (Claude): v4.1 + OLat / NLIinit / BiBCE arms; Commit 4 software identity and receipt binding",''')

rep('''B0_REUSE_IGNORE_TASK = ("code_sha256", "loss")''', '''SOFT_ENV_KEYS = ("torch", "cuda", "transformers")
RUN_COMMIT = []  # set in main(); one_task uses it for the Commit 4 resume check
ARM_NUMERIC_KEYS = ("bibce_lambda0_equal", "bibce_four_targets", "bibce_extreme", "olat_K1_zero", "olat_K4_zero", "olat_K4_0.002")
ARM_INIT_KEYS = ("b0_newly_initialised_keys", "spm_sha256", "K4_zero", "K4_0.002")
INPUT_NAMES = ("manifest.csv", "fold_selection.csv", "split_manifest.json")


def soft_env(info):
    return {k: str((info or {}).get(k)) for k in SOFT_ENV_KEYS}


def smoke_binding(out, commit):
    """Digests of this run's SMOKE done.json files (they carry runtime and environment), tying a receipt to one SMOKE run."""
    res = {}
    for g in SMOKE_GROUPS[commit]:
        p = Path(out) / "smoke" / g.replace("+", "plus") / "done.json"
        res[g] = digest(p) if p.exists() else None
    return res


def arm_receipt(out, rev, info, numeric, init):
    return {"status": "pass", "code_sha256": digest(Path(__file__)), "revision": rev,
            "input_hashes": {n: digest(Path(out) / "input_data" / n) for n in INPUT_NAMES},
            "stable_environment": stable_env_identity(info), "software_environment": soft_env(info),
            "smoke_binding": smoke_binding(out, "4"), "numeric": numeric, "init": init}


def arm_receipt_problems(out, rev, info):
    """Empty list = arm_checks.json is complete, passed, and belongs to this code, model, inputs, environment and SMOKE run."""
    p = Path(out) / "arm_checks.json"
    if not p.exists():
        return ["arm_checks.json missing"]
    a, probs = jload(p), []
    if a.get("status") != "pass":
        probs.append("status")
    if a.get("code_sha256") != digest(Path(__file__)):
        probs.append("code_sha256")
    if a.get("revision") != rev:
        probs.append("revision")
    if a.get("input_hashes") != {n: digest(Path(out) / "input_data" / n) for n in INPUT_NAMES}:
        probs.append("input_hashes")
    if a.get("stable_environment") != stable_env_identity(info) or a.get("software_environment") != soft_env(info):
        probs.append("environment")
    sb = smoke_binding(out, "4")
    if None in sb.values() or a.get("smoke_binding") != sb:
        probs.append("smoke_binding")
    n, i = a.get("numeric"), a.get("init")
    if (not isinstance(n, dict) or n.get("status") != "pass" or any(k not in n for k in ARM_NUMERIC_KEYS)
            or n.get("bibce_lambda0_equal") is not True or len(n.get("bibce_four_targets") or []) != 4
            or not all(isinstance(n.get(k), dict) for k in ("olat_K1_zero", "olat_K4_zero", "olat_K4_0.002"))
            or n["olat_K1_zero"].get("max_abs_logit_diff") != 0.0):
        probs.append("numeric")
    if (not isinstance(i, dict) or i.get("status") != "pass" or any(k not in i for k in ARM_INIT_KEYS)
            or not isinstance(i.get("spm_sha256"), dict) or not i["spm_sha256"].get("b0")
            or i["spm_sha256"].get("b0") != i["spm_sha256"].get("nli")
            or not all(isinstance(i.get(k), dict) and i[k].get("global_rng_restored") is True for k in ("K4_zero", "K4_0.002"))
            or not (isinstance(i["K4_zero"].get("max_abs_prob_diff"), (int, float)) and i["K4_zero"]["max_abs_prob_diff"] <= 1e-5)):
        probs.append("init")
    return probs


def env_mismatched_tasks(out, groups, soft_now):
    """Restored done.json files of these groups trained under other torch / cuda / transformers than soft_now."""
    bad = []
    for g in groups:
        for d in sorted((Path(out) / "tasks" / g.replace("+", "plus")).glob("seed*/fold*/done.json")):
            env = soft_env(jload(d).get("environment"))
            if env != soft_now:
                bad.append("%s %s" % (d.parent.relative_to(Path(out) / "tasks"), env))
    return bad


B0_REUSE_IGNORE_TASK = ("code_sha256", "loss")''')

rep('''            die("Existing done.json identity mismatch or prediction corruption: " + str(done_file))
        print("[SKIP authenticated]", group, seed, fold, flush=True)''',
    '''            die("Existing done.json identity mismatch or prediction corruption: " + str(done_file))
        if RUN_COMMIT and RUN_COMMIT[0] == "4" and soft_env(d.get("environment")) != soft_env(envinfo()):
            die("Restored task %s seed %d fold %d was trained under %s, this worker runs %s: start a fresh output or keep the "
                "environment" % (group, seed, fold, soft_env(d.get("environment")), soft_env(envinfo())))
        print("[SKIP authenticated]", group, seed, fold, flush=True)''')

rep('''    args=parser.parse_args()''', '''    args=parser.parse_args()
    RUN_COMMIT[:] = [args.commit]''')

rep('''            ac = {"numeric": arm_numeric, "init": arm_init_checks(args.model, rev, tokenizer, original, splits),
                  "code_sha256": digest(Path(__file__)), "status": "pass"}''',
    '''            ac = arm_receipt(out, rev, info, arm_numeric, arm_init_checks(args.model, rev, tokenizer, original, splits))''')

rep('''        # v5: arm checks and OLat1 = B0 must come from this code; OLat1 = B0 is recomputed read-only from the SMOKE artifacts
        ap = out/"arm_checks.json"
        if not ap.exists() or jload(ap).get("status") != "pass" or jload(ap).get("code_sha256") != digest(Path(__file__)):
            die("arm_checks.json missing, failed or from other code: rerun --stage smoke --group all")
        if olat1_equivalence(out, write=False)["status"] != "pass":
            die("OLat1 = B0 check fails on the SMOKE artifacts: formal queue blocked")''',
    '''        # v5.1: worker software = SMOKE software; receipt complete and bound to this SMOKE run; OLat1 = B0 recomputed read-only;
        # restored tasks must come from the same software. All of this happens before the first task.
        drift = sorted(g for g in SMOKE_GROUPS["4"] if soft_env(jload(out/"smoke"/g/"done.json").get("environment")) != soft_env(info))
        if drift:
            die("torch / cuda / transformers of this worker %s differ from the SMOKE run (%s): rerun --stage smoke --group all "
                "in this environment first" % (soft_env(info), drift))
        probs = arm_receipt_problems(out, rev, info)
        if probs:
            die("arm_checks.json does not validate %s: rerun --stage smoke --group all" % probs)
        if olat1_equivalence(out, write=False)["status"] != "pass":
            die("OLat1 = B0 check fails on the SMOKE artifacts: formal queue blocked")
        stale = env_mismatched_tasks(out, ["B0"] + COMMITS["4"], soft_env(info))
        if stale:
            die("%d restored task(s) were trained under other torch / cuda / transformers, e.g. %s: start a fresh output "
                "or rerun in their environment" % (len(stale), stale[:5]))''')

open(dst, "w", encoding="utf-8").write(s)
print("ok", dst)
