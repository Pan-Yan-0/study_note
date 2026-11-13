"""用真实冻结包回归训练目标、Commit 5 复用和继续运行。python test_a1.py <解压的训练包>"""
import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from a1_data import ARMS, B2, GROUPS, base, load_targets, preflight, target_counts, task_config, training_targets
from run_a1 import RECEIPT, bindings, gate, pack, prepare_run, queue, restore, verify_tasks
from score_a1 import collect_task, difference
from train_a1 import b2_reuse, numeric_smoke, output_hashes, training_dataset


class A1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = preflight(BUNDLE)
        cls.archive = cls.inputs.bundle / "baseline"
        cls.info = base.jload(cls.archive / "smoke/done.json")["environment"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_all_fold_targets(self):
        changed = 0
        for fold in base.FOLDS:
            reference, _ = training_targets(self.inputs, B2, fold)
            rl, _ = training_targets(self.inputs, "B2-RL", fold)
            sl, probs = training_targets(self.inputs, "B2-SL", fold)
            ls, smooth = training_targets(self.inputs, "B2-LS", fold)
            self.assertEqual(reference, sl)
            self.assertEqual(reference, ls)
            changed += sum(a != b for a, b in zip(reference, rl, strict=True))
            for i, row in enumerate(self.inputs.train[fold]):
                if row["source_group"] == "original":
                    self.assertEqual(reference[i], rl[i])
                    self.assertEqual(probs[i], smooth[i])
                    self.assertEqual(sum(probs[i]), 1.0)
                    self.assertEqual(probs[i][reference[i]], 1.0)
                else:
                    teacher = self.inputs.teacher[fold][row["sample_id"]]
                    self.assertEqual(rl[i], base.L2I[teacher["teacher"]])
                    self.assertEqual(probs[i], teacher["p"])
                    self.assertEqual(smooth[i], [0.925 if j == reference[i] else 0.025 for j in range(4)])
        self.assertEqual(changed, 1794)

    def test_batch_collation_keeps_hard_and_soft_targets_aligned(self):
        import torch
        from transformers import BertTokenizer

        vocab = self.root / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n", encoding="utf-8")
        tokenizer = BertTokenizer(str(vocab))
        n_orig = len(self.inputs.splits["0"]["fit"])
        indices = [n_orig + 11, 0, n_orig + 10, 1]
        control, hard_collator = training_dataset(self.inputs, B2, 0, tokenizer)
        hard = hard_collator([control[i] for i in indices])
        for group in ("B2-RL", "B2-SL", "B2-LS"):
            dataset, collator = training_dataset(self.inputs, group, 0, tokenizer)
            batch = collator([dataset[i] for i in indices])
            for name in hard.keys() - {"labels"}:
                self.assertTrue(torch.equal(hard[name], batch[name]))
            y, targets = training_targets(self.inputs, group, 0)
            self.assertEqual(batch["labels"].tolist(), [y[i] for i in indices])
            if targets:
                expected = torch.tensor([targets[i] for i in indices], dtype=torch.float32)
                self.assertTrue(torch.equal(batch["soft_targets"], expected))
            else:
                self.assertNotIn("soft_targets", batch)

    def test_reject_stale_teacher(self):
        fake = self.root / "teacher_targets.jsonl"
        fake.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            load_targets(fake, self.inputs.train)

    def test_one_hot_update(self):
        self.assertEqual(numeric_smoke("cpu")["actual_updates_each"], 2)

    def simulated_smoke(self):
        # 以真实 Commit 5 概率为测试夹具，测试复用与身份门槛，不代表 A1 实机 SMOKE。
        out = self.root / "run"
        out.mkdir()
        identity = prepare_run(self.inputs, out, self.info)
        archive = self.archive / "smoke"
        old_smoke = base.jload(archive / "smoke.json")
        for group in GROUPS:
            dest = out / "smoke" / group
            shutil.copytree(archive, dest)
            cfg = task_config(self.inputs, group, 42, 0, self.info, 2 if group == B2 else 6)
            task_hash = base.canonical_json_hash(cfg)
            done = base.jload(dest / "done.json")
            done.update(group=group, task_fingerprint=task_hash, target_counts=target_counts(self.inputs, group, 0))
            base.jsave(dest / "task.json", cfg)
            base.jsave(dest / "done.json", done)
            base.jsave(dest / "smoke.json", {**old_smoke, "group": group, "task_fingerprint": task_hash,
                                            "smoke_batches": cfg["smoke_batches"]})
        base.jsave(out / RECEIPT, {"status": "pass", "run_identity": identity, "bindings": bindings(out),
                                  "fold_teacher_sha256": self.inputs.fold_teacher_hashes,
                                  "one_hot_equivalence": {"loss_max_abs": 0, "grad_norm_max_abs": 0,
                                                          "parameter_max_abs": 0, "actual_updates_each": 2},
                                  "b2": b2_reuse(self.inputs, out, self.info)})
        return out

    def test_reuse_and_fallback(self):
        out = self.simulated_smoke()
        self.assertTrue(gate(self.inputs, out, self.info))
        self.assertEqual(len(queue(True)), 45)
        self.assertEqual(len(queue(False)), 60)
        for reuse in (True, False):
            jobs = queue(reuse)
            self.assertEqual(len(jobs), len(set(jobs[::2]) | set(jobs[1::2])))
            self.assertEqual(set(g for g, _, _ in jobs), set(ARMS if reuse else GROUPS))
        dest = out / "smoke" / B2
        path = dest / "predictions.csv"
        raw = path.read_bytes()
        rows = base.csvrows(path)
        rows[0]["p_E"] = "0.00001"
        base.csvsave(path, rows, list(rows[0]))
        done = base.jload(dest / "done.json")
        base.jsave(dest / "done.json", {**done, **output_hashes(dest)})
        self.assertFalse(b2_reuse(self.inputs, out, self.info)["reuse"])
        with self.assertRaisesRegex(RuntimeError, "回执内容已改变"):
            gate(self.inputs, out, self.info)
        path.write_bytes(raw)

    def test_reject_changed_software_and_fold_hash(self):
        out = self.simulated_smoke()
        with self.assertRaisesRegex(RuntimeError, "环境身份已改变"):
            gate(self.inputs, out, {**self.info, "torch": "other"})
        inputs = copy.copy(self.inputs)
        inputs.fold_teacher_hashes = {**inputs.fold_teacher_hashes, 3: "bad"}
        with self.assertRaisesRegex(RuntimeError, "教师目标身份"):
            gate(inputs, out, self.info)

    def test_restore_validates_before_moving_tasks(self):
        out = self.simulated_smoke()
        src = self.root / "old"
        src.mkdir()
        shutil.copyfile(out / "run_identity.json", src / "run_identity.json")
        g, s, f = queue(True)[0]
        task = base.task_path(src, g, s, f)
        task.mkdir(parents=True)
        cfg = task_config(self.inputs, g, s, f, self.info)
        base.jsave(task / "task.json", {**cfg, "teacher_fold_sha256": "old"})
        validate = lambda path: verify_tasks(self.inputs, path, self.info, queue(True))
        with self.assertRaisesRegex(RuntimeError, "旧任务身份改变"):
            restore(src, out, validate)
        self.assertFalse((out / "tasks").exists())
        base.jsave(task / "task.json", cfg)
        archive = self.root / "resume.zip"
        pack(src, archive)
        restore(archive, out, validate)
        self.assertEqual(base.jload(base.task_path(out, g, s, f) / "task.json"), cfg)

    def test_all_archived_predictions_rebuild(self):
        counts = []
        for seed in base.SEEDS:
            for fold in base.FOLDS:
                test, rec = collect_task(self.inputs, self.root, self.info, B2, seed, fold, True)
                counts.append(len(test))
                self.assertTrue(rec["reused"])
        self.assertEqual(sum(counts), 3042 * 3)

    def test_interval_uses_unrounded_lower_bound(self):
        import numpy as np

        a = np.array([[0.8, 0.8, 0.8]] * 120)
        b = np.array([[0.7, 0.7, 0.7]] * 120)
        a[:2] = 0.69
        boot = {"a": a, "b": b}
        points = {"a": a[-1], "b": b[-1]}
        report = difference(boot, points, "a", "b", 3, True)
        self.assertAlmostEqual(report["level"], 0.9833333333333333)
        self.assertLess(report["interval"][0], 0.0)
        self.assertFalse(report["improvement"])


if __name__ == "__main__":
    BUNDLE = Path(sys.argv.pop(1))
    unittest.main(verbosity=2)
