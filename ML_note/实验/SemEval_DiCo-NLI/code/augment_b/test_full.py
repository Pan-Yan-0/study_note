"""全量 +1（方案 B）的 CPU 集成测试：假的生成输出走一遍 主轮 → 合并 → 补充规划 → 补充轮 → 再合并 → 第二次 P1 → 放行。

  DICO_OFF=<官方仓库> DICO_PILOT_PKG=<v6 试点包> DICO_PILOT_OUT=<v6 试点处理结果> python -m unittest test_full -v
  在全量批次包里：DICO_OFF=<官方仓库> DICO_FULL=<全量包目录> python -m unittest test_full -v
    （试点包、试点结果从全量包的 pilot_v6/、pilot_v6_result/ 读，并检查主轮包能逐字节重建）
"""
import csv, hashlib, json, os, shutil, subprocess, sys, tempfile, unittest
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import process_outputs as po  # noqa: E402
import merge_full  # noqa: E402
import plan_supplement  # noqa: E402
import supplement  # noqa: E402

OFF = os.environ.get("DICO_OFF", "")
FULL = os.environ.get("DICO_FULL", "")
PILOT_PKG = os.environ.get("DICO_PILOT_PKG", f"{FULL}/pilot_v6" if FULL else "")
PILOT_OUT = os.environ.get("DICO_PILOT_OUT", f"{FULL}/pilot_v6_result" if FULL else "")
LETTER = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "F", "BACKWARD_ENTAILMENT": "B", "NEGATIVE_OTHER": "O"}


def fake_return(pkg, ret, inject=None):
    """每批一次合格输出；inject = {batch_id: {位置: (text1, text2)}} 用来放进重复的对。"""
    if os.path.exists(ret): shutil.rmtree(ret)
    log = []
    for x in sorted(os.listdir(f"{pkg}/batches")):
        b = po.load(f"{pkg}/batches/{x}"); bid = b["batch_id"]
        pairs = [{"text1": f"zq{bid}x{i} pier", "text2": f"zq{bid}x{i} by water", "label": l} for i, l in enumerate(b["ref_labels"])]
        for i, (t1, t2) in (inject or {}).get(bid, {}).items(): pairs[i].update(text1=t1, text2=t2)
        raw = json.dumps({"batch_id": bid, "pairs": pairs})
        os.makedirs(f"{ret}/outputs/{bid}")
        with open(f"{ret}/outputs/{bid}/attempt_1.txt", "w", encoding="utf-8") as f: f.write(raw)
        log.append({"batch_id": bid, "attempt": 1, "agent_id": f"a-{bid}", "start": "2026-11-07T00:00:00Z", "end": "2026-11-07T00:01:00Z",
                    "requested_model": "gpt-6-luna", "visible_model": "工具未返回", "prompt_sha256": b["prompt_sha256"],
                    "status": "ok", "reason": "", "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()})
    with open(f"{ret}/dispatch_log.jsonl", "w", encoding="utf-8") as f:
        for e in log: f.write(json.dumps(e, ensure_ascii=False) + "\n")


class Full(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OFF and os.path.isdir(OFF) and os.path.isdir(PILOT_PKG) and os.path.isdir(PILOT_OUT)):
            raise RuntimeError("需要 DICO_OFF，以及 DICO_FULL 或 DICO_PILOT_PKG + DICO_PILOT_OUT（测试不会静默跳过）")
        cls.t = tempfile.mkdtemp()
        cls.main = f"{cls.t}/main"
        subprocess.run([sys.executable, f"{HERE}/build_full.py", "main", OFF, PILOT_PKG, cls.main], check=True, capture_output=True)

    def test_main_package(self):
        man = po.load(f"{self.main}/MANIFEST.json")
        parents = [p for x in sorted(os.listdir(f"{self.main}/batches")) for p in po.load(f"{self.main}/batches/{x}")["parents"]]
        fams = [p["family_id"] for p in parents]
        pilot = {p["family_id"] for x in os.listdir(f"{PILOT_PKG}/batches") for p in po.load(f"{PILOT_PKG}/batches/{x}")["parents"]}
        self.assertEqual(len(fams), len(set(fams)))           # 每个家族一次
        self.assertFalse(set(fams) & pilot)                     # 不用试点用过的
        self.assertEqual(len(fams), 1489)
        for x in sorted(os.listdir(f"{self.main}/batches")):    # 一批一个签名，可用折 = 签名
            b = po.load(f"{self.main}/batches/{x}")
            self.assertLessEqual(len(b["parents"]), 14)
            self.assertEqual({tuple(p["usable_folds"]) for p in b["parents"]}, {tuple(b["usable_folds"])})
        if FULL:  # 在全量包里：重建的主轮包和包里的逐字节相同
            for rel, h in po.load(f"{FULL}/main/MANIFEST.json")["files"].items():
                self.assertEqual(po.sha_file(f"{self.main}/{rel}"), h, rel)
            self.assertEqual(man["files"], po.load(f"{FULL}/main/MANIFEST.json")["files"])

    def test_pipeline(self):
        t = self.t
        # 主轮：放一个和试点重复的对，合并时应当被跨阶段去重去掉
        pm = [r for r in po.read_csv(f"{PILOT_OUT}/manifest.csv") if r["direction"] == "original"]
        b1 = po.load(f"{self.main}/batches/main_b001.json")
        i = b1["ref_labels"].index(pm[0]["label"])
        fake_return(self.main, f"{t}/ret_main", {"main_b001": {i: (pm[0]["text1"], pm[0]["text2"])}})
        po.build(self.main, f"{t}/ret_main", OFF, f"{t}/out_main", "test-main")
        specs = {"pilot": (PILOT_PKG, PILOT_OUT), "main": (self.main, f"{t}/out_main")}
        g1 = merge_full.merge(OFF, f"{t}/m1", specs)
        r1 = po.load(f"{t}/m1/report.json")
        self.assertEqual(r1["problems"], [])
        self.assertEqual(r1["stats"]["dropped_cross_stage"], {"main": 1})
        self.assertEqual(r1["checks"]["fold_quota"], "fail")     # 主轮之后有缺口（只差数量的家族不用、试点有损失）
        self.assertFalse(g1["ready_for_training"])
        # 补充第 1 轮：按实际缺口规划，建包，假输出，再合并
        plan = plan_supplement.main(OFF, PILOT_PKG, f"{t}/m1", 1, None, f"{t}/plan1.json", [PILOT_PKG, self.main])
        self.assertEqual(plan["status"], "ok")
        self.assertLessEqual(plan["ledger"]["rows_issued"], supplement.BUDGET * plan["budget_denominator_rows"])
        self.assertLessEqual(max(plan["ledger"]["uses"].values()), supplement.MAX_USES)
        subprocess.run([sys.executable, f"{HERE}/build_full.py", "supp", OFF, PILOT_PKG, f"{t}/plan1.json", f"{t}/supp1"],
                       check=True, capture_output=True)
        fake_return(f"{t}/supp1", f"{t}/ret_s1")
        po.build(f"{t}/supp1", f"{t}/ret_s1", OFF, f"{t}/out_s1", "test-supp1")
        specs["supp1"] = (f"{t}/supp1", f"{t}/out_s1")
        merge_full.merge(OFF, f"{t}/m2", specs)
        r2 = po.load(f"{t}/m2/report.json")
        self.assertEqual(r2["problems"], [])
        self.assertEqual(r2["checks"]["fold_quota"], "pass", r2["stats"]["fold_quota_short"])
        # 逐折选用：行数恰好等于该折 fit 的四类行数，同一对的方向一起选，选中的折在可用折里
        split = po.load(f"{PILOT_PKG}/split_manifest.json")["folds"]
        lab = {r["instance_id"]: r["label"] for r in po.read_csv(f"{OFF}/final_data/train/dico_nli_train_track1_participant_labeled.csv")}
        man = {r["sample_id"]: r for r in po.read_csv(f"{t}/m2/manifest.csv")}
        sel = po.read_csv(f"{t}/m2/fold_selection.csv")
        by_fold = defaultdict(Counter); pair_rows = defaultdict(set)
        for s in sel:
            r = man[s["sample_id"]]; k = int(s["fold"])
            self.assertIn(str(k), r["usable_folds"].split(";"))
            by_fold[k][r["label"]] += 1; pair_rows[(k, r["pair_uid"])].add(r["direction"])
        for k in range(5):
            self.assertEqual(by_fold[k], Counter(lab[i] for i in split[str(k)]["fit"]))
        for (k, u), ds in pair_rows.items():
            n = sum(1 for r in man.values() if r["pair_uid"] == u)
            self.assertEqual(len(ds), n)                         # 一对的所有方向都在同一折里被选
        self.assertTrue(all(r["direct_parent_family_id"] and r["ref_index"] for r in man.values()))
        conc = po.load(f"{t}/m2/concentration.json")
        self.assertLessEqual(max(map(int, conc["dispatch_per_family"])), supplement.MAX_USES)
        # 第二次 P1：只抽新增、被选用的对；两人都照生成标签标 → 放行；之后改动逐折选用 → 失效
        key = po.load(f"{t}/m2/review_key.json")["items"]
        self.assertEqual(len(key), 60)
        self.assertTrue(all(v["stage"] != "pilot" for v in key.values()))
        sel_uids = {s["pair_uid"] for s in sel}
        self.assertTrue(all(v["pair_uid"] in sel_uids for v in key.values()))
        for who in ("a", "b"):
            with open(f"{t}/{who}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["review_id", "label"])
                w.writerows([(rid, LETTER[v["generated_label_shown_orientation"]]) for rid, v in key.items()])
        po.score(f"{t}/m2", "P1", f"{t}/a.csv", f"{t}/b.csv")
        gate = po.load(f"{t}/m2/gate.json")
        self.assertTrue(gate["ready_for_training"], gate["checks"])
        self.assertIn("fold_selection.csv", gate["bound_to"])
        with open(f"{t}/m2/fold_selection.csv", "a", encoding="utf-8") as f: f.write("0,x,y\n")
        self.assertFalse(po.write_gate(f"{t}/m2")["ready_for_training"])


if __name__ == "__main__":
    unittest.main()
