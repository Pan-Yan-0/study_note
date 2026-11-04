"""CPU 测试：用假的生成输出走一遍 build_pilot → check_format → process_outputs（不调用任何模型）。
用法：DICO_OFF=<官方仓库目录> python -m unittest test_augment_b -v"""
import csv, json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = os.environ.get("DICO_OFF", "")
sys.path.insert(0, HERE)
from check_format import check  # noqa: E402
import process_outputs as po  # noqa: E402


def fake_pairs(batch, tag):
    q, out = batch["quota_pairs"], []
    for lab, n in q.items():
        for i in range(n):
            out.append({"text1": f"zq {tag} {lab[:3].lower()} {i} on a quiet pier", "text2": f"zq {tag} {lab[:3].lower()} {i} near water", "label": lab})
    return out


@unittest.skipUnless(OFF and os.path.isdir(OFF), "需要 DICO_OFF 指向官方仓库")
class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.mkdtemp()
        cls.pkg = f"{cls.t}/pkg"
        subprocess.run([sys.executable, f"{HERE}/build_pilot.py", OFF, cls.pkg], check=True, capture_output=True)
        cls.batches = {x[:-5]: json.load(open(f"{cls.pkg}/batches/{x}")) for x in sorted(os.listdir(f"{cls.pkg}/batches"))}

    def test_package(self):
        self.assertEqual(len(self.batches), 15)
        fam = set()
        for b in self.batches.values():
            self.assertEqual(len({p["family_id"] for p in b["parents"]}), len(b["parents"]))
            fam |= {p["family_id"] for p in b["parents"]}
            for p in b["parents"]:
                self.assertTrue(set(b["usable_folds"]) <= set(p["usable_folds"]))  # 子样本可用折 ⊆ 每个父样本可用折
            self.assertIn(b["batch_id"], b["prompt"])
            self.assertNotIn("{reference_pairs}", b["prompt"])
        self.assertEqual(len(fam), 195)  # 批次之间也不重复用父样本
        # 父样本只在可用折的 fit 里，从不在那些折的 val / test 里
        split = json.load(open(f"{self.pkg}/split_manifest.json"))["folds"]
        for b in self.batches.values():
            for k in b["usable_folds"]:
                fit = set(split[str(k)]["fit"])
                self.assertTrue(all(p["sample_id"] in fit for p in b["parents"]))

    def test_check_format(self):
        b = self.batches["pilot_b01"]
        good = json.dumps({"batch_id": "pilot_b01", "pairs": fake_pairs(b, "x")})
        self.assertEqual(check(b, good)["status"], "ok")
        self.assertTrue(check(b, "```json\n" + good + "\n```")["fence_stripped"])
        bad_count = json.loads(good); bad_count["pairs"].pop()
        self.assertEqual(check(b, json.dumps(bad_count))["status"], "format_fail")
        bad_label = json.loads(good); bad_label["pairs"][0]["label"] = "BACKWARD_ENTAILMENT"
        self.assertEqual(check(b, json.dumps(bad_label))["status"], "format_fail")
        self.assertEqual(check(b, good + " trailing")["status"], "format_fail")
        self.assertEqual(check(b, good.replace("pilot_b01", "pilot_b02"))["status"], "format_fail")

    def test_process_and_score(self):
        ret = f"{self.t}/ret"; out = f"{self.t}/out"
        log = []
        train = list(csv.DictReader(open(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
        for bid, b in self.batches.items():
            os.makedirs(f"{ret}/outputs/{bid}", exist_ok=True)
            pairs = fake_pairs(b, bid)
            if bid == "pilot_b02":  # 各种要被过滤的情况
                pairs[0] = {"text1": "same thing", "text2": "Same  thing", "label": pairs[0]["label"]}
                p0 = b["parents"][0]; pairs[1] = {"text1": p0["text2"], "text2": p0["text1"], "label": pairs[1]["label"]}
                r = train[100]; pairs[2] = {"text1": r["text1"], "text2": r["text2"], "label": pairs[2]["label"]}
                pairs[3] = dict(fake_pairs(self.batches["pilot_b01"], "pilot_b01")[0], label=pairs[3]["label"])
                pairs[4] = {"text1": "w " * 25, "text2": "x", "label": pairs[4]["label"]}
            attempts = []
            if bid == "pilot_b03": attempts = [("format_fail", json.dumps({"batch_id": bid, "pairs": pairs[:-1]})), ("ok", json.dumps({"batch_id": bid, "pairs": pairs}))]
            elif bid == "pilot_b05": attempts = [("format_fail", "not json"), ("format_fail", "{}")]
            elif bid == "pilot_b07": attempts = [("infra_error", ""), ("ok", json.dumps({"batch_id": bid, "pairs": pairs}))]
            else: attempts = [("ok", json.dumps({"batch_id": bid, "pairs": pairs}))]
            for n, (st, raw) in enumerate(attempts, 1):
                if st != "infra_error":
                    open(f"{ret}/outputs/{bid}/attempt_{n}.txt", "w").write(raw)
                log.append({"batch_id": bid, "attempt": n, "agent_id": f"a{bid}{n}", "start": "t0", "end": "t1",
                            "requested_model": "gpt-6-luna", "visible_model": "gpt-6-luna", "prompt_sha256": b["prompt_sha256"],
                            "status": st, "reason": ""})
        with open(f"{ret}/dispatch_log.jsonl", "w") as f:
            for e in log: f.write(json.dumps(e) + "\n")
        st = po.build(self.pkg, ret, OFF, out, "test")
        self.assertEqual(st["problems"], [])
        self.assertEqual(st["failed_batches"], ["pilot_b05"])
        self.assertEqual(st["removed"], {"same_text": 1, "dup_parent": 1, "dup_official": 1, "dup_generated": 1, "length": 1})
        rows = list(csv.DictReader(open(f"{out}/manifest.csv")))
        labs = {(r["pair_uid"], r["direction"]): r["label"] for r in rows}
        for (uid, d), l in labs.items():
            if d == "flipped": self.assertIn(l, ("EQUIVALENCE", "BACKWARD_ENTAILMENT"))
        self.assertFalse(any(r["label"] == "NEGATIVE_OTHER" and r["direction"] == "flipped" for r in rows))
        self.assertTrue(all(r["usable_folds"] for r in rows))
        sheet = list(csv.DictReader(open(f"{out}/review_sheet.csv")))
        self.assertEqual(len(sheet), 60)
        self.assertNotIn("label_generated", sheet[0])
        key = json.load(open(f"{out}/review_key.json"))["items"]
        # P1：两人都照生成标签标 → 不一致率 0
        perfect = f"{self.t}/a.csv"
        with open(perfect, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["review_id", "text1", "text2", "label", "note"])
            for s in sheet: w.writerow([s["review_id"], s["text1"], s["text2"], key[s["review_id"]]["generated_label_shown_orientation"], ""])
        res, stop = po.score(f"{out}/review_key.json", "P1", perfect, perfect)
        self.assertFalse(stop)
        # P2：Claude 全标 O，裁决表为空 → E 和单向的不一致率 100%，触发停止
        allo = f"{self.t}/o.csv"; empty = f"{self.t}/e.csv"
        with open(allo, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["review_id", "label"])
            for s in sheet: w.writerow([s["review_id"], "O"])
        open(empty, "w").write("review_id,label\n")
        res, stop = po.score(f"{out}/review_key.json", "P2", allo, empty)
        self.assertTrue(stop)
        # P2 裁决清单：Claude 全对时只剩 10 个抽查项；裁决把它们都改成生成标签，不触发停止
        adj = f"{self.t}/adj.csv"
        po.adjudication_list(f"{out}/review_key.json", perfect, adj)
        items = list(csv.DictReader(open(adj)))
        self.assertEqual(len(items), 10); self.assertTrue(all(r["why"] == "抽查" for r in items))
        po.adjudication_list(f"{out}/review_key.json", allo, adj)
        self.assertEqual(sum(r["why"] == "不一致" for r in csv.DictReader(open(adj))), 40)


if __name__ == "__main__":
    unittest.main()
