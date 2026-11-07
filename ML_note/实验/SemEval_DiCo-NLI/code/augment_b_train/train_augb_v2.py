#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DiCo-NLI AugB Haiku +1: frozen-data verification, Kaggle SMOKE, 45 CE jobs.

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
GROUPS = ("B0", "B1-Haiku", "step+1")
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
          "seed_note": "init=seed*1000+fold; per-epoch fresh shuffle RNG (not archived C0 RNG stream)"}


def die(message):
    raise RuntimeError(message)


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


def make_rows(group, fold, original_lookup, selected_by_fold, splits):
    fit = [dict(original_lookup[s]) for s in splits["folds"][str(fold)]["fit"]]
    for r in fit:
        r["row_id"] = "orig:" + r["instance_id"]
        r["source_group"] = "original"
    if group == "B0":
        return fit
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
    total_steps = math.ceil(len(train)/CONFIG["batch_size"]) * num_epochs
    cfg = {"group": group, "fold": fold, "seed": seed, "model": model,
           "revision": revision, "train_rows": len(train), "val_rows": len(validate), "test_rows": len(test),
           "steps": total_steps, "epochs": num_epochs, "smoke": smoke, "config": CONFIG,
           "data_sha256": {n: digest(Path(out)/"input_data"/n) for n in ("manifest.csv", "fold_selection.csv", "split_manifest.json")},
           "official_data_sha256": digest(Path(out)/"official_reference.sha256") if (Path(out)/"official_reference.sha256").exists() else None,
           "code_sha256": digest(Path(__file__)),
           "stable_environment": stable_env_identity(envinfo()),
           "seed_policy": CONFIG["seed_note"]}
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
                loss = F.cross_entropy(logits.float(), y)
            if not bool(torch.isfinite(loss)):
                die("Nonfinite training loss: " + group)
            status, grad_norm = apply_update(loss, optimizer, sched, scaler, list(net.parameters()))
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
    if not checkpoint.exists():
        die("No best checkpoint")
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    checkpoint.unlink()
    if smoke:
        if updates < 1:
            die("SMOKE produced zero actual optimizer updates")
        jsave(dest/"smoke.json", {"group":group,"status":"pass","updates":updates,"skips":skips,
                                   "code_sha256": digest(Path(__file__)), "revision": revision,
                                   "stable_environment": stable_env_identity(envinfo()),
                                   "input_hashes": cfg["data_sha256"],
                                   "validation_wf1":best,"epochs":epoch_log,"elapsed_sec":round(time.time()-started,3)})
        del net
        torch.cuda.empty_cache()
        return
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
            "predictions_sha256":digest(dest/"predictions.csv"),
            "val_predictions_sha256":digest(dest/"val_predictions.csv"),
            "seed_policy": CONFIG["seed_note"], "inference_precision": "fp32_no_autocast",
            "runtime_sec":round(time.time()-started,3),"environment":envinfo()}
    jsave(done_file, done)
    print("[DONE]",group,seed,fold,"val=",round(best,5),"test_sym=",round(done["test_sym_wf1"],5),flush=True)
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
    args=parser.parse_args()
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
        finish(out,root,original,fam_of,args.bootstraps)
        return
    import torch
    if not torch.cuda.is_available():
        die("No GPU. Upload/run on Kaggle with GPU T4 selected")
    if "T4" not in torch.cuda.get_device_name(0).upper():
        die("Frozen AugB training environment requires Kaggle Tesla T4; got " + torch.cuda.get_device_name(0))
    rev=pin_model(args.model,out,args.model_revision)
    info=envinfo()
    record_environment(out, info, rev)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=None if Path(args.model).exists() else rev)
    if args.stage=="smoke":
        for group in GROUPS if args.group=="all" else [args.group]:
            one_task(group,42,0,original,selected,splits,fam_of,out,tokenizer,args.model,rev,smoke=True)
        print("[SMOKE PASS] Three groups (or selected group) completed two optimizer steps, checkpoint restore and validation")
        return
    # Formal jobs only if all three SMOKE groups were validated in this run's environment.
    for g in GROUPS:
        sp=out/"smoke"/g.replace("+","plus")/"smoke.json"
        if not sp.exists():
            die("SMOKE not passed for "+g+". Run --stage smoke --group all first")
        sr=jload(sp)
        current_hashes={n:digest(out/"input_data"/n) for n in ("manifest.csv","fold_selection.csv","split_manifest.json")}
        if (sr.get("status") != "pass" or sr.get("code_sha256") != digest(Path(__file__))
            or sr.get("revision") != rev or sr.get("input_hashes") != current_hashes
            or sr.get("stable_environment") != stable_env_identity(info)):
            die("SMOKE identity stale after code/model/data/GPU change; rerun --stage smoke")
    targets=[g for g in GROUPS if args.group=="all" or g==args.group]
    seeds=[args.seed] if args.seed is not None else SEEDS
    folds=[args.fold] if args.fold is not None else FOLDS
    for seed in seeds:
        for fold in folds:
            for group in targets:
                one_task(group,seed,fold,original,selected,splits,fam_of,out,tokenizer,args.model,rev)
    if args.seed is None and args.fold is None and args.group=="all":
        finish(out,root,original,fam_of,args.bootstraps)


if __name__=="__main__":
    try: main()
    except Exception as err:
        import traceback
        traceback.print_exc()
        sys.exit(1)
