"""Build train_augb_v4.py from train_augb_v3.py by exact, once-only replacements."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)


rep('"""DiCo-NLI AugB v3 (Claude, 11-07):',
    '"""DiCo-NLI AugB v4.1 (Claude, 11-10): v3 + two loss groups for Commit 3 (Focal, Dice). The CE line is unchanged;\n'
    'new losses are computed in fp32 outside autocast. SMOKE checks that the v4 B0 SMOKE reproduces the archived\n'
    'Commit 1 B0 SMOKE (config and every saved probability); if not, B0 is rerun in the formal queue.\n\n'
    'v3 docstring follows.\nDiCo-NLI AugB v3 (Claude, 11-07):')

rep('''              **{"G-" + p: {"kind": "mixgen", "arm": "G-" + p} for p in POINTS}}''',
    '''              **{"G-" + p: {"kind": "mixgen", "arm": "G-" + p} for p in POINTS},
              "Focal": {"kind": "orig", "loss": "focal_softmax_a1_g2", "gamma": 2.0},
              "Dice": {"kind": "orig", "loss": "soft_dice_macro4_batch", "eps": 1e-6}}''')

rep('''           "2": ["B2-Haiku", "B4-Haiku"] + ["G-" + p for p in POINTS]}
SMOKE_GROUPS = {"1": ["B0", "W2", "step+4", "Dup-O1"], "2": ["B4-Haiku", "G-O1"]}''',
    '''           "2": ["B2-Haiku", "B4-Haiku"] + ["G-" + p for p in POINTS],
           "3": ["Focal", "Dice"]}  # v4: B0 is added only if the B0 reuse check fails
SMOKE_GROUPS = {"1": ["B0", "W2", "step+4", "Dup-O1"], "2": ["B4-Haiku", "G-O1"], "3": ["B0", "Focal", "Dice"]}''')

rep('''          "code_version": "v3 (Claude): v2 training core + new groups",''',
    '''          "code_version": "v4.1 (Claude): v3 + Focal / Dice loss branches",
          "loss_variants": "ce (unchanged line, inside autocast); focal_softmax_a1_g2: mean(-(1-p_t)^2 log p_t); "
                           "soft_dice_macro4_batch: 1 - mean_c (2 sum p_c y_c + 1e-6) / (sum p_c + sum y_c + 1e-6), "
                           "batch-level, all four classes incl. absent; both fp32 outside autocast",''')

rep('''def die(message):
    raise RuntimeError(message)
''', '''def die(message):
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


B0_REUSE_IGNORE_TASK = ("code_sha256", "loss")  # loss is checked separately below (missing = ce; both must be ce)
B0_REUSE_ENV = {"torch": "2.11.0+cu128", "cuda": "12.8", "transformers": "4.57.6"}  # archived Commit 1 values
B0_REUSE_IGNORE_CONFIG = ("code_version", "loss_variants")


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
''')

rep('''    spec = GROUP_SPEC[group]
    lam = float(spec.get("lambda_O", 1.0))''', '''    spec = GROUP_SPEC[group]
    loss_kind = spec.get("loss", "ce")
    lam = float(spec.get("lambda_O", 1.0))''')

rep('''           "class_weights": ([round(w, 10) for w in class_w] if lam != 1.0 else None)}''',
    '''           "class_weights": ([round(w, 10) for w in class_w] if lam != 1.0 else None), "loss": loss_kind}''')

rep('''            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(**batch).logits
                if lam == 1.0:
                    loss = F.cross_entropy(logits.float(), y)  # unchanged v2 / C0 branch
                else:
                    wv = torch.tensor(class_w, device=logits.device, dtype=torch.float32)
                    loss = (wv[y] * F.cross_entropy(logits.float(), y, reduction="none")).mean()''',
    '''            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(**batch).logits
                if loss_kind == "ce":
                    if lam == 1.0:
                        loss = F.cross_entropy(logits.float(), y)  # unchanged v2 / C0 branch
                    else:
                        wv = torch.tensor(class_w, device=logits.device, dtype=torch.float32)
                        loss = (wv[y] * F.cross_entropy(logits.float(), y, reduction="none")).mean()
            if loss_kind != "ce":
                loss = new_loss(loss_kind, spec, logits, y)  # v4: fp32, outside autocast''')

rep('''    if smoke and updates < 1:
        die("SMOKE produced zero actual optimizer updates")''',
    '''    if smoke and updates < 1:
        die("SMOKE produced zero actual optimizer updates")
    if smoke and loss_kind != "ce" and updates != 2:
        die("v4 SMOKE for %s needs exactly 2 actual optimizer updates, got %d (skips %d)" % (group, updates, skips))''')

rep('''    parser.add_argument("--commit",choices=["1","2"],''', '''    parser.add_argument("--commit",choices=["1","2","3"],''')

rep('''    if args.stage=="preflight":
        return
    # Bind formal''', '''    if args.commit == "3":
        if not (bundle / "data_v4" / "c1_smoke_B0" / "predictions.csv").exists():
            die("Commit 3 needs data_v4/c1_smoke_B0 (archived Commit 1 B0 SMOKE) for the B0 reuse check")
        try:
            import torch  # noqa: F401
            print("[PREFLIGHT V4 LOSS CHECKS PASS] cpu", json.dumps(loss_checks("cpu")), flush=True)
        except ImportError:
            print("[PREFLIGHT V4] torch not importable here; loss checks run again on GPU in SMOKE", flush=True)
    if args.stage=="preflight":
        return
    # Bind formal''')

rep('''        for group in SMOKE_GROUPS[args.commit] if args.group=="all" else [args.group]:
            one_task(group,42,0,original,selected,splits,fam_of,out,tokenizer,args.model,rev,smoke=True)''',
    '''        if args.commit == "3":
            lc = loss_checks("cuda")
            jsave(out/"loss_checks.json", lc)
            print("[SMOKE V4 LOSS CHECKS PASS] cuda", json.dumps(lc), flush=True)
        for group in SMOKE_GROUPS[args.commit] if args.group=="all" else [args.group]:
            one_task(group,42,0,original,selected,splits,fam_of,out,tokenizer,args.model,rev,smoke=True)
        if args.commit == "3" and args.group == "all":
            br = b0_reuse_check(bundle, out)
            print("[B0 REUSE CHECK]", br["status"], br["differences"], br["compared"], flush=True)''')

rep('''    targets=[g for g in COMMITS[args.commit] if args.group=="all" or g==args.group]''',
    '''    queue_groups = list(COMMITS[args.commit])
    if args.commit == "3":
        # v4.1: recompute from the current archive and the SMOKE artifacts verified above (read-only); never trust a stale file
        if not (out/"smoke"/"B0"/"done.json").exists():
            die("B0 SMOKE missing: run --stage smoke --group all first")
        br = b0_reuse_check(bundle, out, write=False)
        bp = out/"b0_reuse.json"
        if bp.exists() and jload(bp).get("status") != br["status"]:
            print("[B0 REUSE] note: stored b0_reuse.json disagrees with the recomputed decision; using the recomputed one", flush=True)
        if br["status"] != "pass":
            queue_groups = ["B0"] + queue_groups
            print("[B0 REUSE] failed in SMOKE: B0 is rerun under v4 (15 extra tasks)", flush=True)
        else:
            print("[B0 REUSE] passed in SMOKE: Commit 1 B0 is reused; not rerun", flush=True)
    targets=[g for g in queue_groups if args.group=="all" or g==args.group]''')

rep('''    updates, skips = 0, 0''', '''    updates, skips = 0, 0
    grad_trace = []  # v4.1: per-step status and unscaled, pre-clip global grad norm (SMOKE only)''')
rep('''            status, grad_norm = apply_update(loss, optimizer, sched, scaler, list(net.parameters()))''',
    '''            status, grad_norm = apply_update(loss, optimizer, sched, scaler, list(net.parameters()))
            if smoke:
                grad_trace.append({"epoch": ep + 1, "batch": step, "status": status, "grad_norm_unscaled_preclip": (grad_norm if math.isfinite(grad_norm) else None),
                                   "loss": float(loss.detach())})''')
rep('''        jsave(dest/"smoke.json", {"group":group,"status":"pass","updates":updates,"skips":skips,''',
    '''        jsave(dest/"smoke.json", {"group":group,"status":"pass","updates":updates,"skips":skips,"grad_trace":grad_trace,''')

open(dst, "w", encoding="utf-8").write(s)
print("ok", dst)
