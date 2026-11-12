"""v6（Commit 5：B2-NLIinit 对 B2-Haiku）的 CPU 测试，不需要 torch。

用真实训练包和官方数据跑 main()：torch / transformers 换成只报版本的假模块，pin_model 和 one_task 换成记录调用的桩，
所以只检查到"每个 worker 会按什么顺序跑哪些任务"为止，不训练。

  python3 -I test_v6.py <训练包目录> <官方仓库目录>
"""
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, types, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = Path(sys.argv.pop(1)).resolve()
OFFICIAL = Path(sys.argv.pop(1)).resolve()
SCRIPT = BUNDLE / "train_augb_v6.py" if (BUNDLE / "train_augb_v6.py").exists() else HERE / "train_augb_v6.py"
REV = "a0484667b22365f84929a935b5e50a51f71f2c03"
INFO = {"python": "3.12", "torch": "2.11.0+cu128", "transformers": "4.57.6", "sklearn": "1.6", "cuda": "12.8",
        "gpu": "Tesla T4", "precision": "fp16", "platform": "linux"}


NLI_REPORT = {"global_rng_restored": True, "nli_init": {"model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
              "revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "tensors_loaded": 200, "parameters_loaded": 278809344,
              "kept_from_b0": ["classifier.bias", "classifier.weight"], "forward_config_differences_kept_b0": {}}}  # as the Commit 4 SMOKE wrote it


def fake_modules(info):
    torch = types.ModuleType("torch")
    torch.__version__ = info["torch"]
    torch.version = types.SimpleNamespace(cuda=info["cuda"])
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i=0: info["gpu"])
    tr = types.ModuleType("transformers")
    tr.__version__ = info["transformers"]
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: "tokenizer")
    return {"torch": torch, "transformers": tr}


def load(name="tv6"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Commit5(unittest.TestCase):
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

    def run_main(self, out, stage, commit, worker=0, workers=1, info=INFO, extra=(), patch=None):
        tv = load()
        if patch:
            patch(tv)
        sys.modules.update(fake_modules(info))
        tv.pin_model = lambda *a, **k: REV
        tv.envinfo = lambda: dict(info)
        tv.one_task = lambda g, s, f, *a, smoke=False, **k: self.calls.append((g, s, f, smoke))
        argv = ["train_augb_v6.py", "--stage", stage, "--bundle", str(BUNDLE), "--official", str(OFFICIAL), "--out", str(out),
                "--model-revision", REV, "--worker", str(worker), "--workers", str(workers)] + (["--commit", commit] if commit else []) + list(extra)
        old = sys.argv
        sys.argv = argv
        try:
            tv.main()
        finally:
            sys.argv = old
        return tv

    def fake_smoke(self, out, commit, info=INFO):
        """SMOKE artifacts as the real SMOKE writes them (after a preflight has copied input_data)."""
        tv = load()
        (Path(out) / "input_data").mkdir(parents=True, exist_ok=True)
        for n in tv.INPUT_NAMES:  # as the full stage copies them
            shutil.copy2(BUNDLE / "data" / n, Path(out) / "input_data" / n)
        hashes = {n: sha(Path(out) / "input_data" / n) for n in tv.INPUT_NAMES}
        tv.jsave(Path(out) / "environment.json", {"stable": {"gpu": info["gpu"], "precision": info["precision"]}, "model_revision": REV})
        for g in tv.SMOKE_GROUPS[commit]:
            d = Path(out) / "smoke" / g.replace("+", "plus")
            d.mkdir(parents=True, exist_ok=True)
            tv.jsave(d / "smoke.json", {"status": "pass", "code_sha256": sha(SCRIPT), "revision": REV, "input_hashes": hashes,
                                        "stable_environment": {"gpu": info["gpu"], "precision": info["precision"]}})
            if commit == "5":
                rec = tv.jload(d / "smoke.json")
                rec["arm_report"] = None if g == "B2-Haiku" else dict(NLI_REPORT, nli_init=dict(NLI_REPORT["nli_init"]))
                tv.jsave(d / "smoke.json", rec)
            tv.jsave(d / "done.json", {"group": g, "environment": info})
        self.calls.clear()

    def test_rebuild_chain_is_byte_exact(self):
        for mk, src, want in (("make_v5.py", "train_augb_v4.py", "train_augb_v5.py"), ("make_v6.py", "train_augb_v5.py", "train_augb_v6.py")):
            dst = self.tmp / want
            subprocess.run([sys.executable, "-I", str(SCRIPT.parent / mk), str(SCRIPT.parent / src), str(dst)], check=True, capture_output=True)
            self.assertEqual(sha(dst), sha(SCRIPT.parent / want), want)

    def test_spec_is_b2_plus_nli_init_only(self):
        tv = load()
        b2, nli, arm = tv.GROUP_SPEC["B2-Haiku"], tv.GROUP_SPEC["NLIinit"], tv.GROUP_SPEC["B2-NLIinit"]
        init = {k: v for k, v in nli.items() if k.startswith("init")}
        self.assertEqual(arm, {**b2, **init})
        self.assertEqual(set(nli) - set(init), {"kind"})
        self.assertEqual(tv.COMMITS["5"], ["B2-Haiku", "B2-NLIinit"])
        self.assertEqual(tv.SMOKE_GROUPS["5"], ["B2-Haiku", "B2-NLIinit"])
        self.assertIn(("B2-NLIinit", "B2-Haiku"), tv.PAIRED)

    def test_earlier_queues_unchanged(self):
        spec = importlib.util.spec_from_file_location("tv5", SCRIPT.parent / "train_augb_v5.py")
        tv5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv5)
        tv = load()
        for c in ("1", "2", "3", "4"):
            self.assertEqual(tv.COMMITS[c], tv5.COMMITS[c])
            self.assertEqual(tv.SMOKE_GROUPS[c], tv5.SMOKE_GROUPS[c])
        for g, s in tv5.GROUP_SPEC.items():
            self.assertEqual(tv.GROUP_SPEC[g], s)

    def test_rows_identical_to_b2_on_real_data(self):
        tv = load()
        gen, selected, splits, _ = tv.preflight_generated(BUNDLE, self.tmp)
        original, fam_of, _ = tv.preflight_official(BUNDLE, tv.official_root(str(OFFICIAL)), selected, splits, self.tmp)
        tv.load_v3_data(BUNDLE, "5", splits)
        for k in range(5):
            a = tv.make_rows("B2-NLIinit", k, original, selected, splits)
            b = tv.make_rows("B2-Haiku", k, original, selected, splits)
            c = tv.make_rows("step+2", k, original, selected, splits)
            self.assertEqual(a, b)
            self.assertEqual(len(a), len(c))
            self.assertEqual(len({r["row_id"] for r in a}), len(a))
            self.assertGreater(sum(r["row_id"].startswith("gen:") for r in a), 0)

    def test_formal_queue_is_30_tasks_without_b0(self):
        out = self.tmp / "run"
        self.fake_smoke(out, "5")
        got = []
        for w in (0, 1):
            self.calls.clear()
            self.run_main(out, "full", "5", worker=w, workers=2)
            got.append(list(self.calls))
        allt = got[0] + got[1]
        self.assertEqual(len(allt), 30)
        self.assertEqual(len(set(allt)), 30)
        self.assertEqual({g for g, *_ in allt}, {"B2-Haiku", "B2-NLIinit"})
        self.assertEqual(len(got[0]), 15)
        self.assertFalse(any(s for *_, s in allt))
        self.assertEqual(sorted((s, f) for g, s, f, _ in allt if g == "B2-NLIinit"), [(s, f) for s in (42, 43, 44) for f in range(5)])

    def test_smoke_missing_blocks(self):
        out = self.tmp / "run"
        self.fake_smoke(out, "5")
        shutil.rmtree(out / "smoke" / "B2-NLIinit")
        with self.assertRaises(RuntimeError):
            self.run_main(out, "full", "5")
        self.assertEqual(self.calls, [])

    def test_software_drift_blocks_before_first_task(self):
        out = self.tmp / "run"
        self.fake_smoke(out, "5")
        other = dict(INFO, transformers="4.58.0")
        with self.assertRaisesRegex(RuntimeError, "differ from the SMOKE run"):
            self.run_main(out, "full", "5", info=other)
        self.assertEqual(self.calls, [])

    def test_restored_task_from_other_software_blocks(self):
        out = self.tmp / "run"
        self.fake_smoke(out, "5")
        d = out / "tasks" / "B2-Haiku" / "seed43" / "fold2"
        d.mkdir(parents=True)
        (d / "done.json").write_text(json.dumps({"environment": dict(INFO, torch="2.10.0+cu126")}))
        with self.assertRaisesRegex(RuntimeError, "restored task"):
            self.run_main(out, "full", "5")
        self.assertEqual(self.calls, [])

    def test_bad_nli_smoke_record_blocks(self):
        for name, edit in (("missing", lambda r: r.pop("arm_report")), ("199 tensors", lambda r: r["arm_report"]["nli_init"].update(tensors_loaded=199)),
                           ("other revision", lambda r: r["arm_report"]["nli_init"].update(revision="0" * 40)),
                           ("kept pooler", lambda r: r["arm_report"]["nli_init"].update(kept_from_b0=["classifier.bias", "classifier.weight", "pooler.dense.bias"])),
                           ("rng", lambda r: r["arm_report"].update(global_rng_restored=False))):
            out = self.tmp / ("run_" + name.replace(" ", "_"))
            self.fake_smoke(out, "5")
            tv = load()
            p = out / "smoke" / "B2-NLIinit" / "smoke.json"
            r = tv.jload(p); edit(r); tv.jsave(p, r)
            with self.assertRaisesRegex(RuntimeError, "init record does not validate"):
                self.run_main(out, "full", "5")
            self.assertEqual(self.calls, [], name)
        out = self.tmp / "run_haiku_arm"
        self.fake_smoke(out, "5")
        p = out / "smoke" / "B2-Haiku" / "smoke.json"
        r = load().jload(p); r["arm_report"] = NLI_REPORT; load().jsave(p, r)
        with self.assertRaisesRegex(RuntimeError, "B2-Haiku arm_report"):
            self.run_main(out, "full", "5")

    def test_commit4_still_requires_arm_receipt(self):
        if not (BUNDLE / "data_v4" / "c1_smoke_B0").exists():
            self.skipTest("bundle has no data_v4")
        out = self.tmp / "run4"
        self.fake_smoke(out, "4")
        stub = lambda tv: setattr(tv, "arm_checks", lambda device: {"status": "pass"})  # the CPU numeric check needs real torch
        with self.assertRaisesRegex(RuntimeError, "arm_checks.json does not validate"):
            self.run_main(out, "full", "4", patch=stub)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
