"""CPU 测试：用假的生成输出走一遍 build_pilot → check_format → process_outputs → 审核计分和放行门槛（不调用任何模型）。

在仓库里：DICO_OFF=<官方仓库目录> python -m unittest test_augment_b -v
在批次包里：DICO_OFF=<官方仓库目录> DICO_PKG=<批次包目录> python -m unittest test_augment_b -v
  （用包内的 inputs/families.json 和 split_manifest.json 重建，检查和包里的文件逐字节相同）
没有 DICO_OFF 时测试直接失败，不会静默跳过。"""
import csv, hashlib, json, os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = os.environ.get("DICO_OFF", "")
PKG = os.environ.get("DICO_PKG", "")
sys.path.insert(0, HERE)
from check_format import check  # noqa: E402
import process_outputs as po  # noqa: E402
import supplement  # noqa: E402


def fake_pairs(batch, tag):
    out = []
    for lab, n in batch["quota_pairs"].items():
        for i in range(n):
            out.append({"text1": f"zq{tag}{lab[:3].lower()}{i} pier", "text2": f"zq{tag}{lab[:3].lower()}{i} by water", "label": lab})
    return out


def write_ann(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["review_id", "label"]); w.writerows(rows)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OFF and os.path.isdir(OFF)):
            raise RuntimeError("需要设置 DICO_OFF 指向官方仓库目录（测试不会静默跳过）")
        cls.t = tempfile.mkdtemp()
        cls.pkg = f"{cls.t}/pkg"
        cmd = [sys.executable, f"{HERE}/build_pilot.py", OFF, cls.pkg]
        if PKG: cmd += ["--families", f"{PKG}/inputs/families.json", "--split-manifest", f"{PKG}/split_manifest.json"]
        subprocess.run(cmd, check=True, capture_output=True)
        cls.batches = {x[:-5]: po.load(f"{cls.pkg}/batches/{x}") for x in sorted(os.listdir(f"{cls.pkg}/batches"))}

    def test_rebuild_matches_package(self):
        if not PKG: self.skipTest("只在批次包里检查重建一致性")
        mine, theirs = po.load(f"{self.pkg}/MANIFEST.json"), po.load(f"{PKG}/MANIFEST.json")
        self.assertEqual(mine["files"], theirs["files"])
        self.assertEqual(mine["builder_sha256"], theirs["builder_sha256"])

    def test_package(self):
        self.assertEqual(len(self.batches), 15)
        fam = set()
        split = po.load(f"{self.pkg}/split_manifest.json")["folds"]
        for b in self.batches.values():
            self.assertEqual(len({p["family_id"] for p in b["parents"]}), len(b["parents"]))
            fam |= {p["family_id"] for p in b["parents"]}
            for p in b["parents"]:
                self.assertTrue(set(b["usable_folds"]) <= set(p["usable_folds"]))
            for k in b["usable_folds"]:  # 父样本只在可用折的 fit 里
                self.assertTrue(all(p["sample_id"] in set(split[str(k)]["fit"]) for p in b["parents"]))
            self.assertIn("real examples from the dataset", b["prompt"])
            mx = max(len(p[k].split()) for p in b["parents"] for k in ("text1", "text2"))  # 长度要求来自本批父样本
            self.assertIn(f"No phrase may have more than {mx} words.", b["prompt"]); self.assertNotIn("{ref_", b["prompt"])
        self.assertEqual(len(fam), 195)
        d = po.load(f"{self.pkg}/debug/debug_b04.json")
        self.assertIn("hand-written examples", d["prompt"]); self.assertNotIn("real examples", d["prompt"])

    def test_check_format(self):
        b = self.batches["pilot_b01"]
        good = json.dumps({"batch_id": "pilot_b01", "pairs": fake_pairs(b, "x")})
        self.assertEqual(check(b, good)["status"], "ok")
        self.assertTrue(check(b, "```json\n" + good + "\n```")["fence_stripped"])
        for raw in ("null", "", "  ", "[]", "{}", good + " trailing", good.replace("pilot_b01", "pilot_b02")):
            self.assertEqual(check(b, raw)["status"], "format_fail", raw[:20])
        x = json.loads(good); x["pairs"].pop(); self.assertEqual(check(b, json.dumps(x))["status"], "format_fail")
        x = json.loads(good); x["pairs"][0]["label"] = "BACKWARD_ENTAILMENT"; self.assertEqual(check(b, json.dumps(x))["status"], "format_fail")

    def make_return(self, name, broken=True):
        ret = f"{self.t}/{name}"
        if os.path.exists(ret): shutil.rmtree(ret)
        train = po.read_csv(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv")
        log = []
        for bid, b in self.batches.items():
            os.makedirs(f"{ret}/outputs/{bid}")
            pairs = fake_pairs(b, bid)
            if broken and bid == "pilot_b02":  # 各种要被过滤的情况
                p0 = b["parents"][0]
                pairs[0] = {"text1": "same thing", "text2": "Same  thing", "label": pairs[0]["label"]}
                pairs[1] = {"text1": p0["text2"], "text2": p0["text1"], "label": pairs[1]["label"]}
                pairs[2] = {"text1": train[100]["text1"], "text2": train[100]["text2"], "label": pairs[2]["label"]}
                pairs[3] = dict(fake_pairs(self.batches["pilot_b01"], "pilot_b01")[0], label=pairs[3]["label"])
                pairs[4] = {"text1": "w " * 25, "text2": "x", "label": pairs[4]["label"]}
                pairs[5] = {"text1": "zq brand new thing", "text2": b["parents"][1]["text2"], "label": pairs[5]["label"]}
            ok = json.dumps({"batch_id": bid, "pairs": pairs})
            if broken and bid == "pilot_b03": attempts = [("format_fail", json.dumps({"batch_id": bid, "pairs": pairs[:-1]})), ("ok", ok)]
            elif broken and bid == "pilot_b05": attempts = [("format_fail", "not json"), ("format_fail", "null")]
            elif broken and bid == "pilot_b07": attempts = [("infra_error", None), ("ok", ok)]
            else: attempts = [("ok", ok)]
            for n, (st, raw) in enumerate(attempts, 1):
                e = {"batch_id": bid, "attempt": n, "agent_id": f"a{bid}{n}", "start": "2026-11-05T00:00:00Z",
                     "end": "2026-11-05T00:01:00Z", "requested_model": "gpt-6-luna", "visible_model": "工具未返回",
                     "prompt_sha256": b["prompt_sha256"], "status": st, "reason": "" if st == "ok" else "测试用失败"}
                if raw is not None:
                    with open(f"{ret}/outputs/{bid}/attempt_{n}.txt", "w", encoding="utf-8") as f: f.write(raw)
                    e["raw_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                if st == "infra_error": e["has_partial_output"] = raw is not None
                log.append(e)
        with open(f"{ret}/dispatch_log.jsonl", "w", encoding="utf-8") as f:
            for e in log: f.write(json.dumps(e) + "\n")
        return ret, log

    def test_filters_and_failures(self):
        ret, _ = self.make_return("ret_broken")
        out = f"{self.t}/out_broken"
        r = po.build(self.pkg, ret, OFF, out, "test")
        self.assertEqual(r["problems"], [])
        self.assertEqual(r["stats"]["failed_batches"], ["pilot_b05"])
        self.assertEqual(r["stats"]["removed"], {"same_text": 1, "dup_parent": 1, "dup_official": 1, "dup_generated": 1,
                                                 "length": 1, "reuse_parent_phrase": 1})
        self.assertEqual(r["checks"]["batch_fail_rate"], "fail")  # 1/15 > 5%
        self.assertFalse(po.load(f"{out}/gate.json")["ready_for_training"])
        rows = po.read_csv(f"{out}/manifest.csv")
        self.assertTrue(all(r["label"] in ("EQUIVALENCE", "BACKWARD_ENTAILMENT") for r in rows if r["direction"] == "flipped"))
        self.assertFalse(any(r["label"] == "NEGATIVE_OTHER" and r["direction"] == "flipped" for r in rows))

    def test_protocol_violations_block(self):
        ret, log = self.make_return("ret_proto", broken=False)
        log[0]["start"] = None                                                   # 缺起止时间
        log[2]["end"] = "2026-11-04T00:00:00Z"                                   # 结束早于开始
        del log[3]["raw_sha256"]                                                 # 原文哈希缺失
        log.append(dict(log[1], attempt=2, status="infra_error", reason="x", has_partial_output=False))  # 合格后又追加
        log[4]["status"] = "format_fail"                                         # 失败但没写原因
        log[4]["reason"] = ""
        with open(f"{ret}/dispatch_log.jsonl", "w", encoding="utf-8") as f:
            for e in log: f.write(json.dumps(e) + "\n")
        r = po.build(self.pkg, ret, OFF, f"{self.t}/out_proto", "test")
        self.assertEqual(r["checks"]["protocol"], "fail")
        ps = " | ".join(r["problems"])
        for frag in ("缺 start", "结束时间早于开始", "原文哈希缺失", "已有合格输出后又追加", "失败没有写原因"):
            self.assertIn(frag, ps)
        self.assertFalse(po.load(f"{self.t}/out_proto/gate.json")["ready_for_training"])

    def test_infra_partial_output(self):
        ret, log = self.make_return("ret_infra", broken=False)
        bid = "pilot_b04"; b = self.batches[bid]
        os.rename(f"{ret}/outputs/{bid}/attempt_1.txt", f"{ret}/outputs/{bid}/attempt_2.txt")
        with open(f"{ret}/outputs/{bid}/attempt_1.txt", "w", encoding="utf-8") as f: f.write('{"batch_id": "pilot_b04", "pai')
        i = next(j for j, e in enumerate(log) if e["batch_id"] == bid)
        ok = dict(log[i], attempt=2)
        log[i] = dict(log[i], status="infra_error", reason="连接中断", has_partial_output=True, raw_sha256="0" * 64)
        log.insert(i + 1, ok)
        with open(f"{ret}/dispatch_log.jsonl", "w", encoding="utf-8") as f:
            for e in log: f.write(json.dumps(e) + "\n")
        r = po.build(self.pkg, ret, OFF, f"{self.t}/out_infra", "test")
        self.assertTrue(any("部分输出的哈希" in p for p in r["problems"]))

    def test_review_and_gate(self):
        ret, _ = self.make_return("ret_clean", broken=False)
        out = f"{self.t}/out_clean"
        r = po.build(self.pkg, ret, OFF, out, "test")
        self.assertTrue(all(v == "pass" for k, v in r["checks"].items() if k != "review"), r["checks"])
        self.assertFalse(po.load(f"{out}/gate.json")["ready_for_training"])  # 审核没做之前不放行
        key = po.load(f"{out}/review_key.json")["items"]
        ids = sorted(key)
        good = [(i, key[i]["generated_label_shown_orientation"]) for i in ids]
        A, B = f"{self.t}/a.csv", f"{self.t}/b.csv"
        write_ann(A, []); write_ann(B, good)  # 空表
        self.assertEqual(po.score(out, "P1", A, B)[0]["status"], "invalid")
        write_ann(A, good[:-1]); self.assertEqual(po.score(out, "P1", A, B)[0]["status"], "invalid")  # 缺一项
        write_ann(A, good + good[:1]); self.assertEqual(po.score(out, "P1", A, B)[0]["status"], "invalid")  # 重复
        write_ann(A, [(i, "maybe") for i, _ in good]); self.assertEqual(po.score(out, "P1", A, B)[0]["status"], "invalid")
        write_ann(A, [(i, "U") for i, _ in good])  # 全部无法确定 → 可判定项为 0 → insufficient
        res, gate = po.score(out, "P1", A, B)
        self.assertEqual(res["status"], "insufficient"); self.assertFalse(gate["ready_for_training"])
        write_ann(A, good)
        res, gate = po.score(out, "P1", A, B)
        self.assertEqual(res["status"], "pass"); self.assertTrue(gate["ready_for_training"])
        # P2：Claude 全标 O → 40 项要裁决 + 10 项抽查；裁决表必须覆盖全部 50 项
        write_ann(A, [(i, "O") for i in ids])
        lst = po.adjudication_list(out, A)
        self.assertEqual(sum(r["why"] == "不一致" for r in lst), 40); self.assertEqual(sum(r["why"] == "抽查" for r in lst), 10)
        write_ann(B, [(r["review_id"], key[r["review_id"]]["generated_label_shown_orientation"]) for r in lst[:-1]])
        self.assertEqual(po.score(out, "P2", A, B)[0]["status"], "invalid")
        write_ann(B, [(r["review_id"], key[r["review_id"]]["generated_label_shown_orientation"]) for r in lst])
        res, gate = po.score(out, "P2", A, B)
        self.assertEqual(res["status"], "pass"); self.assertFalse(gate["ready_for_training"])  # 本试点只认 P1
        write_ann(B, [(r["review_id"], "O") for r in lst])
        self.assertEqual(po.score(out, "P2", A, B)[0]["status"], "fail")
        # 审核通过后候选被改动 → 审核结果失效
        write_ann(A, good); write_ann(B, good)
        self.assertTrue(po.score(out, "P1", A, B)[1]["ready_for_training"])
        with open(f"{out}/manifest.csv", "a", encoding="utf-8") as f: f.write("x\n")
        self.assertFalse(po.write_gate(out)["ready_for_training"])
        # 同一目录重建：旧审核结果被清掉，回到 pending
        po.build(self.pkg, ret, OFF, out, "test")
        self.assertFalse(os.path.exists(f"{out}/review_result.json"))
        g = po.load(f"{out}/gate.json"); self.assertEqual(g["checks"]["review"], "pending"); self.assertFalse(g["ready_for_training"])

    def test_probe_missing_blocks(self):
        ret, _ = self.make_return("ret_probe", broken=False)
        orig = po.probe
        po.probe = lambda pairs, seed=0: None
        try:
            r = po.build(self.pkg, ret, OFF, f"{self.t}/out_probe", "test")
        finally:
            po.probe = orig
        self.assertEqual(r["checks"]["probe_gap"], "not_run")

    def test_supplement_plan(self):
        sig = {"0134": ["a", "b"], "0124": ["c"], "1234": ["d", "e", "f"], "024": ["g"]}
        kr = {"E": 0.83, "U": 0.9, "O": 0.7}
        deficit = {(0, "E"): 30, (2, "E"): 10, (3, "O"): 5}
        b1, st1, led1 = supplement.plan(deficit, sig, kr, 3000, supplement.new_ledger(), 1)
        b2, st2, _ = supplement.plan(deficit, sig, kr, 3000, supplement.new_ledger(), 1)
        self.assertEqual(st1, "ok"); self.assertEqual(b1, b2)  # 确定性
        self.assertEqual([b["signature"] for b in b1 if b["group"] == "E"][0], "0124")  # 同时覆盖 0、2 折
        self.assertTrue(all(len(b["parent_families"]) <= 13 for b in b1))
        with self.assertRaises(ValueError): supplement.plan(deficit, sig, kr, 3000, led1, 3)   # 不允许第三轮
        with self.assertRaises(ValueError): supplement.plan(deficit, sig, kr, 3000, led1, 1)   # 同一轮不能再来
        with self.assertRaises(ValueError): supplement.plan(deficit, sig, kr, 3000, supplement.new_ledger(), 2)
        # 预算跨轮累计：每轮各 14 行，计划 100 行 → 第一轮 14% 通过，第二轮累计 28% 超过 20%
        small = {(0, "O"): 1}
        _, s1, l1 = supplement.plan(small, sig, {"E": 1, "U": 1, "O": 1}, 100, supplement.new_ledger(), 1)
        _, s2, _ = supplement.plan(small, sig, {"E": 1, "U": 1, "O": 1}, 100, l1, 2)
        self.assertEqual(s1, "ok"); self.assertIn("预算不足", s2)
        # 游标循环：同一签名连续两批取到不同的家族
        big = {k: list("abcdefghijklmnopqrstuvwxyz") for k in ("0134",)}
        bb, _, _ = supplement.plan({(0, "E"): 40}, big, {"E": 1, "U": 1, "O": 1}, 10 ** 6, supplement.new_ledger(), 1)
        self.assertGreaterEqual(len(bb), 2); self.assertFalse(set(bb[0]["parent_families"]) & set(bb[1]["parent_families"]))


if __name__ == "__main__":
    unittest.main()
