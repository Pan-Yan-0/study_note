#!/usr/bin/env python3
"""agents/check_repo_hygiene.py 的回归测试套件。"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import check_repo_hygiene as crh


class TestRepoHygiene(unittest.TestCase):
    def test_self_scan_zero_hits_without_exemption(self):
        """检查脚本与测试脚本自身在无整文件豁免下扫描 0 命中。"""
        here = Path(__file__).resolve().parent
        for name in ("check_repo_hygiene.py", "test_repo_hygiene.py"):
            text = (here / name).read_text(encoding="utf-8")
            self.assertEqual(crh.scan_text(name, text), [])

    def test_context_lv_prevents_false_positive_and_catches_leak(self):
        """允许 NLP 笔记中的正常缩写（如 Lv et al.），拦截带数字或私有字段上下文的缩写。"""
        ok_text = "As shown in Lv et al. (2024), the Lv projection layer is linear.\n"
        self.assertEqual(crh.scan_text("ML_note/论文阅读/a.md", ok_text), [])

        bad_lv_num = "当前处于 " + "Lv" + "9，距离下一级还差 84。\n"
        self.assertEqual(len(crh.scan_text("README.md", bad_lv_num)), 1)

        bad_lv_ctx = "查询存" + "档里的 Lv 字段。\n"
        self.assertEqual(len(crh.scan_text("agents/doc.md", bad_lv_ctx)), 1)

    def test_exact_allowlist_emails_and_paths(self):
        """白名单精确到匹配值：放行公开 noreply 邮箱和 /kaggle/working，拦截私人邮箱与容器路径。"""
        allowed = (
            "Co-authored-by: Codex <noreply@openai.com>\n"
            "Author: Pan Yan <112330427+Pan-Yan-0@users.noreply.github.com>\n"
            "输出保存在 /kaggle/working/out.zip 和 /mnt/workspace/venv。\n"
        )
        self.assertEqual(crh.scan_text("AGENTS.md", allowed), [])

        bad_email = "联系邮箱：student_private" + "@" + "example.edu.cn\n"
        self.assertEqual(len(crh.scan_text("README.md", bad_email)), 1)

        bad_path = "参见 [结果](file:///" + "home/antigravity/Pan-Yan-0/study_note/a.md)\n"
        self.assertEqual(len(crh.scan_text("agents/分工与纠错机制_v2.md", bad_path)), 1)

    def test_path_ownership_rules(self):
        """单工作线 + 共用索引通过；跨工作线（含 SemEval 与 论文研究/ 混交）触发告警/错误。"""
        issues, _ = crh.check_ownership([
            "README.md",
            "ML_note/实验结果汇总.md",
            "ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md",
        ])
        self.assertEqual(issues, [])

        # SemEval 主线与 论文研究/ 混在同一提交应被判定为跨线
        issues_cross, _ = crh.check_ownership([
            "ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md",
            "ML_note/实验/SemEval_DiCo-NLI/论文研究/reading.md",
        ])
        self.assertEqual(len(issues_cross), 1)

        # SemEval 与 infra (AGENTS.md) 混在同一提交应被判定为跨线
        issues_infra, _ = crh.check_ownership([
            "ML_note/实验/SemEval_DiCo-NLI/README.md",
            "AGENTS.md",
        ])
        self.assertEqual(len(issues_infra), 1)

    def test_multi_commit_range_catches_reverted_leak_and_scans_shared_indexes(self):
        """一次 push 包含多个 commit 时，前序 commit 混入 README.md 或 agents/ 的隐私即使在后序 commit 删掉也必须被拦下。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
            subprocess.run(["git", "-C", tmp, "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", tmp, "config", "user.email", "noreply@anthropic.com"], check=True)

            (repo / "README.md").write_text("# Clean Root\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "base"], check=True)
            base_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            # Commit 1: 在共用索引 README.md 和 agents/ 中混入私有字段与容器路径
            (repo / "agents").mkdir()
            (repo / "README.md").write_text("# Root\n未结算" + "经验" + "值 15\n", encoding="utf-8")
            (repo / "agents" / "note.md").write_text("路径 /" + "home/antigravity/secret.md\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md", "agents/note.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c1 leak"], check=True)

            # Commit 2: 将两处违规内容删回干净状态，模拟“后序 commit 掩盖前序 commit”
            (repo / "README.md").write_text("# Clean Root\n", encoding="utf-8")
            (repo / "agents" / "note.md").write_text("干净归档\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md", "agents/note.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c2 mask"], check=True)

            rc = crh.main(["--repo", tmp, "--commit-range", f"{base_rev}..HEAD"])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
