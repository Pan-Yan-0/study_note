"""v5.1 Commit 4 正式队列门槛的 CPU 测试（不需要 torch）：arm_checks.json 的完整性和绑定、续跑任务的软件版本。

  python3 -I test_v5_receipts.py [train_augb_v5.py 的路径]
"""
import importlib.util, json, os, shutil, sys, tempfile, unittest
from pathlib import Path

SCRIPT = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1].endswith(".py") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_augb_v5.py")
spec = importlib.util.spec_from_file_location("tv5", SCRIPT)
tv5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv5)

INFO = {"gpu": "Tesla T4", "precision": "fp16", "torch": "2.11.0+cu128", "cuda": "12.8", "transformers": "4.57.6",
        "python": "3.12", "sklearn": "1.6", "platform": "linux"}
NUMERIC = {"status": "pass", "bibce_lambda0_equal": True, "bibce_four_targets": [[0.1, 0.1]] * 4, "bibce_extreme": 3.0,
           "olat_K1_zero": {"max_abs_logit_diff": 0.0}, "olat_K4_zero": {"max_abs_logit_diff": 1e-7},
           "olat_K4_0.002": {"max_abs_logit_diff": 1e-3}}
INIT = {"status": "pass", "b0_newly_initialised_keys": ["classifier.bias", "classifier.weight", "pooler.dense.bias", "pooler.dense.weight"],
        "spm_sha256": {"b0": "ab" * 32, "nli": "ab" * 32},
        "K4_zero": {"max_abs_prob_diff": 1.2e-8, "global_rng_restored": True},
        "K4_0.002": {"max_abs_prob_diff": 4e-5, "global_rng_restored": True}}


def make_run(root, tag, info=INFO):
    out = Path(root)
    (out / "input_data").mkdir(parents=True)
    for n in tv5.INPUT_NAMES:
        (out / "input_data" / n).write_text("data " + n)
    for g in tv5.SMOKE_GROUPS["4"]:
        d = out / "smoke" / g
        d.mkdir(parents=True)
        tv5.jsave(d / "done.json", {"group": g, "runtime_sec": tag, "environment": info})
    return out


class Receipt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = make_run(self.tmp / "run1", 1.0)
        tv5.jsave(self.out / "arm_checks.json", tv5.arm_receipt(self.out, "rev1", INFO, NUMERIC, INIT))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def probs(self, info=INFO, rev="rev1"):
        return tv5.arm_receipt_problems(self.out, rev, info)

    def edit(self, f):
        a = tv5.jload(self.out / "arm_checks.json"); f(a); tv5.jsave(self.out / "arm_checks.json", a)

    def test_good_receipt_passes(self):
        self.assertEqual(self.probs(), [])

    def test_truncated_receipt_fails(self):
        self.edit(lambda a: (a.pop("numeric"), a.pop("init")))
        self.assertEqual(sorted(self.probs()), ["init", "numeric"])
        self.edit(lambda a: a.update(numeric={"status": "pass"}, init={"status": "pass"}))
        self.assertEqual(sorted(self.probs()), ["init", "numeric"])

    def test_receipt_from_another_run_fails(self):
        other = make_run(self.tmp / "run2", 2.0)
        tv5.jsave(other / "arm_checks.json", tv5.arm_receipt(other, "rev1", INFO, NUMERIC, INIT))
        shutil.copy(other / "arm_checks.json", self.out / "arm_checks.json")
        self.assertEqual(self.probs(), ["smoke_binding"])

    def test_smoke_rerun_invalidates_receipt(self):
        tv5.jsave(self.out / "smoke" / "OLat4" / "done.json", {"group": "OLat4", "runtime_sec": 9.0, "environment": INFO})
        self.assertEqual(self.probs(), ["smoke_binding"])
        (self.out / "smoke" / "BiBCE" / "done.json").unlink()
        self.assertIn("smoke_binding", self.probs())

    def test_identity_fields(self):
        self.edit(lambda a: a.update(code_sha256="0" * 64))
        self.assertEqual(self.probs(), ["code_sha256"])
        self.setUp()
        self.assertEqual(self.probs(rev="rev2"), ["revision"])
        self.assertEqual(self.probs(info={**INFO, "torch": "2.12.0+cu128"}), ["environment"])
        self.assertEqual(self.probs(info={**INFO, "gpu": "Tesla P100"}), ["environment"])
        (self.out / "input_data" / "manifest.csv").write_text("changed")
        self.assertEqual(self.probs(), ["input_hashes"])

    def test_sub_receipt_contents(self):
        self.edit(lambda a: a["numeric"]["olat_K1_zero"].update(max_abs_logit_diff=1e-7))
        self.assertEqual(self.probs(), ["numeric"])
        self.setUp()
        self.edit(lambda a: a["init"]["spm_sha256"].update(nli="cd" * 32))
        self.assertEqual(self.probs(), ["init"])
        self.setUp()
        self.edit(lambda a: a["init"]["K4_zero"].update(max_abs_prob_diff=1e-4))
        self.assertEqual(self.probs(), ["init"])
        self.setUp()
        self.edit(lambda a: a["init"]["K4_0.002"].update(global_rng_restored=False))
        self.assertEqual(self.probs(), ["init"])
        self.setUp()
        self.edit(lambda a: a.update(status="fail"))
        self.assertEqual(self.probs(), ["status"])

    def test_missing_receipt(self):
        (self.out / "arm_checks.json").unlink()
        self.assertEqual(self.probs(), ["arm_checks.json missing"])


class ResumedTasks(unittest.TestCase):
    def test_env_mismatched_tasks(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for g, s, k, torch_v in (("OLat4", 42, 0, "2.11.0+cu128"), ("BiBCE", 43, 2, "2.12.0+cu128"), ("B0", 44, 4, "2.11.0+cu128")):
                d = tmp / "tasks" / g / ("seed%d" % s) / ("fold%d" % k)
                d.mkdir(parents=True)
                tv5.jsave(d / "done.json", {"environment": {**INFO, "torch": torch_v}})
            bad = tv5.env_mismatched_tasks(tmp, ["B0"] + tv5.COMMITS["4"], tv5.soft_env(INFO))
            self.assertEqual(len(bad), 1)
            self.assertTrue(bad[0].startswith(os.path.join("BiBCE", "seed43", "fold2")))
            self.assertEqual(tv5.env_mismatched_tasks(tmp, ["NLIinit"], tv5.soft_env(INFO)), [])
            self.assertEqual(tv5.soft_env(None), {"torch": "None", "cuda": "None", "transformers": "None"})
        finally:
            shutil.rmtree(tmp)

    def test_commit4_groups(self):
        self.assertEqual(tv5.COMMITS["4"], ["OLat4", "NLIinit", "BiBCE"])
        self.assertEqual(tv5.SMOKE_GROUPS["4"], ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"])
        self.assertIn("arm_variants", tv5.B0_REUSE_IGNORE_CONFIG)
        self.assertEqual(tv5.GROUP_SPEC["B0"], {"kind": "orig"})


if __name__ == "__main__":
    unittest.main()
