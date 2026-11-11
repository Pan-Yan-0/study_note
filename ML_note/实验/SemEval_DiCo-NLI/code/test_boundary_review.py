"""boundary_review.py 的 score 和标注表检查（合成答案，不读真实答案文件）。python3 -I code/test_boundary_review.py"""
import csv, importlib.util, json, os, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("br", f"{HERE}/boundary_review.py"); br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
E, F, B, O = "EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"
HEAD = ["review_id", "text1", "text2", "label", "o_subtype", "phenomena", "evidence"]


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(HEAD); w.writerows(rows)


class Score(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        # S1：10 个家族金标 E；S3：10 个家族金标 O；S2-O→EFB 两个
        items = {f"R{n:03d}": {"stratum": "S1", "gold": E, "qwen4b_majority": F} for n in range(1, 11)}
        items |= {f"R{n:03d}": {"stratum": "S3", "gold": O, "qwen4b_majority": O} for n in range(11, 21)}
        items |= {f"R{n:03d}": {"stratum": "S2-O→EFB", "gold": O, "qwen4b_majority": E} for n in (21, 22)}
        json.dump({"items": items}, open(f"{self.d}/review_key.json", "w"))
        # S1：6 个两人都判 F（一致但不等于金标）、2 个都判 E、1 个两人不一致、1 个有 U；S3：两人都判 O，标 SIM / REL
        a, b = [], []
        for n in range(1, 11):
            la, lb = ("F", "F") if n <= 6 else ("E", "E") if n <= 8 else ("E", "F") if n == 9 else ("U", "E")
            a.append([f"R{n:03d}", "x", "y", la, "", "MOD", "e"]); b.append([f"R{n:03d}", "x", "y", lb, "", "MOD;SYN", "e"])
        for n in range(11, 23):
            a.append([f"R{n:03d}", "x", "y", "O", "SIM", "", "e"]); b.append([f"R{n:03d}", "x", "y", "O", "REL", "NUM", "e"])
        write(f"{self.d}/A.csv", a); write(f"{self.d}/B.csv", b)

    def test_counts_and_rules(self):
        br.score(self.d, f"{self.d}/A.csv", f"{self.d}/B.csv")
        r = json.load(open(f"{self.d}/review_result.json", encoding="utf-8"))
        self.assertEqual(r["by_stratum"]["S1"], {"一致但不等于金标": 6, "一致且等于金标": 2, "两人不一致": 1, "U": 1})
        self.assertEqual(r["by_stratum"]["S3"], {"一致且等于金标": 10})
        self.assertEqual(r["by_stratum"]["S2"], {"一致且等于金标": 2})
        self.assertEqual(r["R1"]["a1"], [6, 9]); self.assertEqual(r["R1"]["a3"], [0, 10]); self.assertEqual(r["R1"]["c1"], [2, 9])
        self.assertEqual(r["R1"]["读法"], "顽固错误多是对照参考卡也复现不了的标法")
        self.assertEqual(r["R2"]["MOD"], {"families": 10, "一致且等于金标": 2, "候选": False})   # 只数 S1 ∪ S2 里两人都标的
        self.assertNotIn("SYN", r["R2"]); self.assertNotIn("NUM", r["R2"])                       # NUM 只有 B 标
        self.assertEqual(r["R3_o_subtype"], {"S2": {"SIM": 2, "REL": 2}, "S3": {"SIM": 10, "REL": 10}})
        self.assertEqual(r["U_by_stratum"]["S1"], [1, 10])

    def _run(self, s1, s3, extra=()):
        """s1 / s3：每个家族 (两人的共同标签, 金标)；extra：附加的 S2 家族。返回 review_result。"""
        d = tempfile.mkdtemp(); items, a = {}, []
        for n, (st, (lab, gold), ph) in enumerate([("S1", x, "MOD") for x in s1] + [("S3", x, "") for x in s3] + list(extra), 1):
            rid = f"R{n:03d}"; items[rid] = {"stratum": st, "gold": gold, "qwen4b_majority": O}
            a.append([rid, "x", "y", lab, "SIM" if lab == "O" else "", ph, "e"])
        json.dump({"items": items}, open(f"{d}/review_key.json", "w")); write(f"{d}/A.csv", a); write(f"{d}/B.csv", a)
        br.score(d, f"{d}/A.csv", f"{d}/B.csv")
        return json.load(open(f"{d}/review_result.json", encoding="utf-8"))

    def test_r1_exact_30pp_boundary(self):
        # Codex 11-12 的用例：S1 7/10 一致但≠金标、3/10 一致且=金标，S3 4/10；差正好 30 个百分点，应触发第一条
        r = self._run([("F", E)] * 7 + [("E", E)] * 3, [("F", E)] * 4 + [("E", E)] * 6)
        self.assertEqual((r["R1"]["a1"], r["R1"]["a3"]), ([7, 10], [4, 10]))
        self.assertEqual(r["R1"]["读法"], "顽固错误多是对照参考卡也复现不了的标法")
        # 7/10 对 5/12：差约 28.3 个百分点，不触发；c1 = 3/10 < 50%，只报告
        r = self._run([("F", E)] * 7 + [("E", E)] * 3, [("F", E)] * 5 + [("E", E)] * 7)
        self.assertEqual(r["R1"]["读法"], "只报告")

    def test_r1_half_boundaries(self):
        # a1 = 5/10、a3 = 2/10：a1 正好 50%、差正好 30 个百分点，触发第一条
        r = self._run([("F", E)] * 5 + [("E", E)] * 5, [("F", E)] * 2 + [("E", E)] * 8)
        self.assertEqual(r["R1"]["读法"], "顽固错误多是对照参考卡也复现不了的标法")
        # a1 = 5/10、a3 = 3/10：差 20 个百分点，不触发第一条；c1 = 5/10 正好 50%，触发第二条
        r = self._run([("F", E)] * 5 + [("E", E)] * 5, [("F", E)] * 3 + [("E", E)] * 7)
        self.assertEqual(r["R1"]["读法"], "可学但没学到，作为后续改动的目标")

    def test_r2_70pct_boundary(self):
        # S1 里 10 个家族两人都标 MOD，7 个一致且等于金标：正好 70%，列为候选；6 个就不列
        r = self._run([("E", E)] * 7 + [("F", E)] * 3, [])
        self.assertEqual(r["R2"]["MOD"], {"families": 10, "一致且等于金标": 7, "候选": True})
        r = self._run([("E", E)] * 6 + [("F", E)] * 4, [])
        self.assertFalse(r["R2"]["MOD"]["候选"])

    def test_sheet_checks(self):
        ids = [f"R{n:03d}" for n in range(1, 3)]
        bad = [[["R001", "x", "y", "O", "", "", "e"], ["R002", "x", "y", "E", "", "", "e"]],        # O 缺子类
               [["R001", "x", "y", "E", "SIM", "", "e"], ["R002", "x", "y", "E", "", "", "e"]],     # 非 O 填子类
               [["R001", "x", "y", "E", "", "FOO", "e"], ["R002", "x", "y", "E", "", "", "e"]],     # 现象标签非法
               [["R001", "x", "y", "E", "", "", ""], ["R002", "x", "y", "E", "", "", "e"]],         # 缺依据
               [["R001", "x", "y", "E", "", "", "e"]],                                             # 缺行
               [["R001", "x", "y", "X", "", "", "e"], ["R002", "x", "y", "E", "", "", "e"]]]        # 标签非法
        for rows in bad:
            write(f"{self.d}/bad.csv", rows)
            with self.assertRaises(ValueError): br.read_ann(f"{self.d}/bad.csv", ids)
        write(f"{self.d}/ok.csv", [["R001", "x", "y", "o", "sim", "mod; num", "e"], ["R002", "x", "y", "u", "", "", "e"]])
        self.assertEqual(br.read_ann(f"{self.d}/ok.csv", ids)["R001"], {"label": O, "sub": "SIM", "ph": {"MOD", "NUM"}})


if __name__ == "__main__":
    unittest.main()
