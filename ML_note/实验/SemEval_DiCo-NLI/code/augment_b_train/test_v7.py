"""v7（Commit 6：DEC / DEC-H 对 NLIinit）的 CPU 测试，不需要 torch。

用真实训练包和官方数据跑 main()：torch / transformers 换成只报版本的假模块，pin_model 和 one_task 换成记录调用的桩，
dec_checks（要真 torch）换成桩，所以只检查到"每个 worker 会按什么顺序跑哪些任务"为止，不训练。
训练路径里要 torch 的部分（BCE、方向头、DEC-H 载入）由 SMOKE 和 Codex 的 torch CPU 复核覆盖。

  python3 -I test_v7.py <训练包目录> <官方仓库目录>
"""
import hashlib, importlib.util, json, shutil, subprocess, sys, tempfile, types, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = Path(sys.argv.pop(1)).resolve()
OFFICIAL = Path(sys.argv.pop(1)).resolve()
SCRIPT = BUNDLE / "train_augb_v7.py" if (BUNDLE / "train_augb_v7.py").exists() else HERE / "train_augb_v7.py"
REV = "a0484667b22365f84929a935b5e50a51f71f159d"  # the real B0 revision: the control identity compares it with Commit 4
INFO = {"python": "3.12", "torch": "2.11.0+cu128", "transformers": "4.57.6", "sklearn": "1.6", "cuda": "12.8",
        "gpu": "Tesla T4", "precision": "fp16", "platform": "linux"}
NLI = {"model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
       "tensors_loaded": 200, "parameters_loaded": 278809344, "kept_from_b0": ["classifier.bias", "classifier.weight"],
       "forward_config_differences_kept_b0": {}}
NUMERIC = {"status": "pass", "bce_targets_EFBO": [0.7, 0.7], "bce_extreme": 2500.0,
           "recombination": {"max_abs_log_norm": 0.0, "max_abs_log_identity": 0.0, "swap_mirror_exact": True, "cells": 81},
           "decision_rule": {"zero_zero": 3, "pos_zero": 1, "zero_pos": 2, "swap_mirror": True}, "dech_logodds_max_abs": 6e-8,
           "install": {"dec_deterministic": True, "dec_rng_untouched": True, "dech_copy_exact": True, "dech_prob_max_abs": 6e-8}}
INIT = {"status": "pass", "DEC-H": {"max_abs_entail_prob_diff": 3e-7}, "DEC": {"finite": True}}


def fake_modules(info):
    torch = types.ModuleType("torch")
    torch.__version__ = info["torch"]
    torch.version = types.SimpleNamespace(cuda=info["cuda"])
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i=0: info["gpu"])
    tr = types.ModuleType("transformers")
    tr.__version__ = info["transformers"]
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: "tokenizer")
    return {"torch": torch, "transformers": tr}


def load(name="tv7"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def real_data(tmp):
    tv = load("tv7_data")
    gen, selected, splits, _ = tv.preflight_generated(BUNDLE, tmp)
    original, fam_of, _ = tv.preflight_official(BUNDLE, tv.official_root(str(OFFICIAL)), selected, splits, tmp)
    tv.load_v3_data(BUNDLE, "6", splits)
    return tv, original, selected, splits


class Commit6(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
        self.calls = []

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(self.tmp)

    def run_main(self, out, stage, commit="6", worker=0, workers=1, info=INFO, bundle=BUNDLE):
        tv = load()
        tv.dec_checks = lambda device: dict(NUMERIC, device=device)  # needs real torch
        sys.modules.update(fake_modules(info))
        tv.pin_model = lambda *a, **k: REV
        tv.envinfo = lambda: dict(info)
        tv.one_task = lambda g, s, f, *a, smoke=False, **k: self.calls.append((g, s, f, smoke))
        argv = ["train_augb_v7.py", "--stage", stage, "--bundle", str(bundle), "--official", str(OFFICIAL), "--out", str(out),
                "--model-revision", REV, "--worker", str(worker), "--workers", str(workers), "--commit", commit]
        old = sys.argv
        sys.argv = argv
        try:
            tv.main()
        finally:
            sys.argv = old
        return tv

    def fake_smoke(self, out, info=INFO, bundle=BUNDLE):
        """Commit 6 SMOKE artifacts as the real SMOKE writes them: NLIinit = the archived Commit 4 SMOKE (only the code hash in
        smoke.json differs, as a real rerun would write it), DEC / DEC-H with their head records, and the bound dec_checks.json."""
        tv, original, selected, splits = real_data(self.tmp / "data")
        out = Path(out)
        (out / "input_data").mkdir(parents=True, exist_ok=True)
        for n in tv.INPUT_NAMES:
            shutil.copy2(bundle / "data" / n, out / "input_data" / n)
        (out / "official_reference.sha256").write_text(sha(OFFICIAL / "final_data/train/dico_nli_train_track1_reference.csv"))
        hashes = {n: sha(out / "input_data" / n) for n in tv.INPUT_NAMES}
        stable = {"gpu": info["gpu"], "precision": info["precision"]}
        tv.jsave(out / "environment.json", {"stable": stable, "model_revision": REV})
        shutil.copytree(bundle / "data_v7" / "c4_smoke_NLIinit", out / "smoke" / "NLIinit")
        p = out / "smoke" / "NLIinit" / "smoke.json"
        r = tv.jload(p); r["code_sha256"] = sha(SCRIPT); tv.jsave(p, r)
        for g, di in (("DEC", {"mode": "linear", "n_out": 1}), ("DEC-H", {"mode": "nli_head", "n_out": 3, "head_equal_to_checkpoint": True})):
            d = out / "smoke" / g
            d.mkdir(parents=True)
            tv.jsave(d / "smoke.json", {"group": g, "status": "pass", "code_sha256": sha(SCRIPT), "revision": REV, "input_hashes": hashes,
                                        "stable_environment": stable, "updates": 2, "skips": 0, "output_path_checked": True,
                                        "v3_data_sha256": dict(tv.V3["hashes"]),
                                        "grad_trace": [{"epoch": 1, "batch": b, "status": "updated", "grad_norm_unscaled_preclip": 3.0, "loss": 0.7} for b in (0, 1)],
                                        "arm_report": {"global_rng_restored": True, "nli_init": dict(NLI), "dec_init": di}})
            tv.jsave(d / "done.json", {"group": g, "environment": info, "actual_optimizer_updates": 2, "skipped_fp16_updates": 0,
                                       "dec_head_change": {"proj.weight": 2e-5, "proj.bias": 1e-5}})
        tv.csvsave(out / "zeroshot_dech.csv", [{"instance_id": "x", "s_ab": "0.1", "s_ba": "-0.2"}], ["instance_id", "s_ab", "s_ba"])
        zs = {"rows": 3042, "sha256": sha(out / "zeroshot_dech.csv")}
        rc = tv.dec_receipt(out, REV, info, NUMERIC, INIT, zs)
        cp = tv.nli_control_problems(bundle, out, original, selected, splits, REV, info)
        rc["control_identity"] = {"status": "pass" if not cp else "fail", "problems": cp, "tasks": 15}
        tv.jsave(out / "dec_checks.json", rc)
        self.calls.clear()
        return tv

    def formal(self, out, **kw):
        got = []
        for w in (0, 1):
            self.calls.clear()
            self.run_main(out, "full", worker=w, workers=2, **kw)
            got.append(list(self.calls))
        return got[0] + got[1]

    # ------------------------------------------------------------------ code identity and specs
    def test_rebuild_chain_is_byte_exact(self):
        for mk, src, want in (("make_v6.py", "train_augb_v5.py", "train_augb_v6.py"), ("make_v7.py", "train_augb_v6.py", "train_augb_v7.py")):
            dst = self.tmp / want
            subprocess.run([sys.executable, "-I", str(SCRIPT.parent / mk), str(SCRIPT.parent / src), str(dst)], check=True, capture_output=True)
            self.assertEqual(sha(dst), sha(SCRIPT.parent / want), want)

    def test_specs_queues_and_config(self):
        tv = load()
        spec = importlib.util.spec_from_file_location("tv6", SCRIPT.parent / "train_augb_v6.py")
        tv6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv6)
        nli = tv.GROUP_SPEC["NLIinit"]
        self.assertEqual(tv.GROUP_SPEC["DEC"], {**nli, "head": "dec", "head_seed_offset": 30000, "loss": "dec_bce"})
        self.assertEqual(tv.GROUP_SPEC["DEC-H"], {**nli, "head": "dec_h", "loss": "dec_bce"})
        self.assertEqual(tv.COMMITS["6"], ["DEC", "DEC-H"])
        self.assertEqual(tv.SMOKE_GROUPS["6"], ["NLIinit", "DEC", "DEC-H"])
        for c in ("1", "2", "3", "4", "5"):
            self.assertEqual(tv.COMMITS[c], tv6.COMMITS[c])
            self.assertEqual(tv.SMOKE_GROUPS[c], tv6.SMOKE_GROUPS[c])
        for g, s in tv6.GROUP_SPEC.items():
            self.assertEqual(tv.GROUP_SPEC[g], s)
        self.assertEqual(tv.PAIRED, tv6.PAIRED)
        diff = {k for k in set(tv.CONFIG) | set(tv6.CONFIG) if tv.CONFIG.get(k) != tv6.CONFIG.get(k)}
        self.assertEqual(diff, {"code_version", "arm_variants"})
        self.assertTrue(diff <= set(tv.B0_REUSE_IGNORE_CONFIG))

    def test_recombination_and_decision_rule(self):
        import numpy as np
        tv = load()
        grid = np.array([-1e4, -40.0, -3.0, -1e-9, 0.0, 1e-9, 0.5, 3.0, 40.0, 1e4])
        A, B = np.meshgrid(grid, grid)
        a, b = A.ravel(), B.ravel()
        lp, lr = tv.dec_logp_np(a, b), tv.dec_logp_np(b, a)
        self.assertTrue(np.isfinite(lp).all())
        self.assertLess(np.abs(np.logaddexp.reduce(lp, axis=1)).max(), 1e-12)
        self.assertLess(np.abs(lp[:, 0] + lp[:, 3] - lp[:, 1] - lp[:, 2]).max(), 1e-9)
        self.assertTrue(np.array_equal(lr[:, [0, 2, 1, 3]], lp))  # (B, A) is the exact F <-> B mirror of (A, B)
        la, lb = tv.dec_labels_np(a, b), tv.dec_labels_np(b, a)
        self.assertTrue(np.array_equal(lb, np.array([0, 2, 1, 3])[la]))
        self.assertEqual(list(tv.dec_labels_np([0.0, 1.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, 1.0, -1.0])), [3, 1, 2, 0, 3])
        away = (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6)
        self.assertTrue(np.array_equal(la[away], lp[away].argmax(axis=1)))  # the rule = four-class argmax off the ties
        sab, sba = 1.3, -0.4
        p = 1 / (1 + np.exp(-np.array([sab, sba])))
        want = [p[0] * p[1], p[0] * (1 - p[1]), (1 - p[0]) * p[1], (1 - p[0]) * (1 - p[1])]
        self.assertTrue(np.allclose(np.exp(tv.dec_logp_np([sab], [sba])[0]), want, atol=1e-15))

    # ------------------------------------------------------------------ control identity on the real archive
    def test_dec_rows_equal_nliinit_and_archive_control_matches(self):
        tv, original, selected, splits = real_data(self.tmp / "d")
        for k in range(5):
            a = tv.make_rows("NLIinit", k, original, selected, splits)
            self.assertEqual(tv.make_rows("DEC", k, original, selected, splits), a)
            self.assertEqual(tv.make_rows("DEC-H", k, original, selected, splits), a)
            self.assertTrue(all(r["row_id"].startswith("orig:") for r in a))
        out = self.tmp / "o"
        (out / "input_data").mkdir(parents=True)
        for n in tv.INPUT_NAMES:
            shutil.copy2(BUNDLE / "data" / n, out / "input_data" / n)
        (out / "official_reference.sha256").write_text(sha(OFFICIAL / "final_data/train/dico_nli_train_track1_reference.csv"))
        self.assertEqual(tv.nli_control_problems(BUNDLE, out, original, selected, splits, REV, INFO), [])
        self.assertTrue(any("revision" in x for x in tv.nli_control_problems(BUNDLE, out, original, selected, splits, "0" * 40, INFO)))
        self.assertTrue(any("done" in x for x in tv.nli_control_problems(BUNDLE, out, original, selected, splits, REV, dict(INFO, transformers="4.58.0"))))
        copy = self.tmp / "bundle_copy"
        shutil.copytree(BUNDLE, copy)
        for name, edit, expect in (("steps", lambda t, d: t.update(steps=t["steps"] - 1), "task.steps"),
                                   ("fingerprint", lambda t, d: d.update(task_fingerprint="0" * 64), "done"),
                                   ("missing", None, "archive files")):
            c = self.tmp / ("bc_" + name)
            shutil.copytree(BUNDLE, c)
            d = c / "data_v7" / "c4_nliinit_tasks" / "seed43" / "fold3"
            if edit is None:
                (d / "done.json").unlink()
            else:
                t, dn = tv.jload(d / "task.json"), tv.jload(d / "done.json")
                edit(t, dn)
                if name == "steps":  # keep the fingerprint consistent so only the steps check can fire
                    dn["task_fingerprint"] = tv.canonical_json_hash(t)
                tv.jsave(d / "task.json", t); tv.jsave(d / "done.json", dn)
            got = tv.nli_control_problems(c, out, original, selected, splits, REV, INFO)
            self.assertEqual(got, ["seed43/fold3 " + expect], name)

    # ------------------------------------------------------------------ formal queue
    def test_formal_queue_30_tasks_when_nliinit_reusable(self):
        out = self.tmp / "run"
        self.fake_smoke(out)
        allt = self.formal(out)
        self.assertEqual(len(allt), 30)
        self.assertEqual(len(set(allt)), 30)
        self.assertEqual({g for g, *_ in allt}, {"DEC", "DEC-H"})
        self.assertFalse(any(s for *_, s in allt))
        for g in ("DEC", "DEC-H"):
            self.assertEqual(sorted((s, f) for gg, s, f, _ in allt if gg == g), [(s, f) for s in (42, 43, 44) for f in range(5)])

    def test_formal_queue_45_tasks_when_smoke_differs(self):
        out = self.tmp / "run"
        tv = self.fake_smoke(out)
        p = out / "smoke" / "NLIinit" / "predictions.csv"
        rows = tv.csvrows(p)
        k = next(c for c in rows[0] if c.startswith("p_"))
        rows[0][k] = "0.123"
        tv.csvsave(p, rows, list(rows[0]))
        allt = self.formal(out)
        self.assertEqual(len(allt), 45)
        self.assertEqual(sum(g == "NLIinit" for g, *_ in allt), 15)

    def test_formal_queue_45_tasks_when_control_identity_fails(self):
        copy = self.tmp / "bundle_copy"
        shutil.copytree(BUNDLE, copy)
        tv = load()
        d = copy / "data_v7" / "c4_nliinit_tasks" / "seed44" / "fold1"
        t = tv.jload(d / "task.json"); t["train_rows"] += 1; tv.jsave(d / "task.json", t)
        out = self.tmp / "run"
        self.fake_smoke(out, bundle=copy)
        allt = []
        for w in (0, 1):
            self.calls.clear()
            self.run_main(out, "full", worker=w, workers=2, bundle=copy)
            allt += self.calls
        self.assertEqual(len(allt), 45)
        self.assertEqual(sum(g == "NLIinit" for g, *_ in allt), 15)

    # ------------------------------------------------------------------ SMOKE gate
    def check_blocks(self, name, edit, pattern="Commit 6 SMOKE records do not validate"):
        out = self.tmp / ("run_" + name)
        tv = self.fake_smoke(out)
        edit(tv, out)
        with self.assertRaisesRegex(RuntimeError, pattern):
            self.run_main(out, "full")
        self.assertEqual(self.calls, [], name)

    def test_smoke_gate_negatives(self):
        def upd(path, f):
            def e(tv, out):
                p = out / path
                r = tv.jload(p); f(r); tv.jsave(p, r)
            return e
        cases = {
            "no_receipt": lambda tv, out: (out / "dec_checks.json").unlink(),
            "dec_head_still": upd("smoke/DEC/done.json", lambda r: r.update(dec_head_change={"proj.weight": 0.0, "proj.bias": 0.0})),
            "dech_not_copied": upd("smoke/DEC-H/smoke.json", lambda r: r["arm_report"]["dec_init"].update(head_equal_to_checkpoint=False)),
            "dec_wrong_mode": upd("smoke/DEC/smoke.json", lambda r: r["arm_report"]["dec_init"].update(mode="nli_head", n_out=3)),
            "dech_one_update": upd("smoke/DEC-H/smoke.json", lambda r: r.update(updates=1, skips=1)),
            "nli_199": upd("smoke/DEC/smoke.json", lambda r: r["arm_report"]["nli_init"].update(tensors_loaded=199)),
            "rebound": upd("smoke/DEC/done.json", lambda r: r.update(runtime_sec=1.0)),  # another SMOKE run than the receipt
            "numeric_key": upd("dec_checks.json", lambda r: r["numeric"].pop("recombination")),
            "tie_rule": upd("dec_checks.json", lambda r: r["numeric"]["decision_rule"].update(zero_zero=0)),
            "init_diff": upd("dec_checks.json", lambda r: r["init"]["DEC-H"].update(max_abs_entail_prob_diff=1e-3)),
            "no_control": upd("dec_checks.json", lambda r: r.pop("control_identity")),
            "zeroshot": lambda tv, out: (out / "zeroshot_dech.csv").write_text("instance_id,s_ab,s_ba\n"),
            "other_code": upd("dec_checks.json", lambda r: r.update(code_sha256="0" * 64)),
        }
        for name, edit in cases.items():
            self.check_blocks(name, edit)

    def test_software_drift_and_restored_tasks_block(self):
        out = self.tmp / "run"
        self.fake_smoke(out)
        with self.assertRaisesRegex(RuntimeError, "differ from the SMOKE run"):
            self.run_main(out, "full", info=dict(INFO, transformers="4.58.0"))
        self.assertEqual(self.calls, [])
        d = out / "tasks" / "DEC" / "seed42" / "fold4"
        d.mkdir(parents=True)
        (d / "done.json").write_text(json.dumps({"environment": dict(INFO, torch="2.10.0+cu126")}))
        with self.assertRaisesRegex(RuntimeError, "restored task"):
            self.run_main(out, "full")
        self.assertEqual(self.calls, [])

    def test_preflight_needs_archived_nliinit_smoke(self):
        copy = self.tmp / "bundle_copy"
        shutil.copytree(BUNDLE, copy)
        shutil.rmtree(copy / "data_v7" / "c4_smoke_NLIinit")
        with self.assertRaisesRegex(RuntimeError, "c4_smoke_NLIinit"):
            self.run_main(self.tmp / "run", "preflight", bundle=copy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
