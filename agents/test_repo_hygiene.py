#!/usr/bin/env python3
"""agents/check_repo_hygiene.py 的回归测试套件（含 v2 边界回归与 v3 任务范围/Drive 版本绑定回执回归）。"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import check_repo_hygiene as crh


def init_temp_git_repo(tmp: str) -> Path:
    repo = Path(tmp)
    subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
    subprocess.run(["git", "-C", tmp, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", tmp, "config", "user.email", "noreply@anthropic.com"], check=True)
    (repo / "README.md").write_text("# Clean Root\n", encoding="utf-8")
    subprocess.run(["git", "-C", tmp, "add", "README.md"], check=True)
    subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "base"], check=True)
    return repo


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

        issues_cross, _ = crh.check_ownership([
            "ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md",
            "ML_note/实验/SemEval_DiCo-NLI/论文研究/reading.md",
        ])
        self.assertEqual(len(issues_cross), 1)

        issues_infra, _ = crh.check_ownership([
            "ML_note/实验/SemEval_DiCo-NLI/README.md",
            "AGENTS.md",
        ])
        self.assertEqual(len(issues_infra), 1)

        issues_dotfiles, _ = crh.check_ownership(
            ["AGENTS.md", "./.github/workflows/repo-hygiene.yml", ".gitignore"],
            expected_line="infra",
        )
        self.assertEqual(issues_dotfiles, [])

    def test_multi_commit_range_catches_reverted_leak_and_scans_shared_indexes(self):
        """一次 push 包含多个 commit 时，前序 commit 混入 README.md 或 agents/ 的隐私即使在后序 commit 删掉也必须被拦下。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            base_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            (repo / "agents").mkdir()
            (repo / "README.md").write_text("# Root\n未结算" + "经验" + "值 15\n", encoding="utf-8")
            (repo / "agents" / "note.md").write_text("路径 /" + "home/antigravity/secret.md\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md", "agents/note.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c1 leak"], check=True)

            (repo / "README.md").write_text("# Clean Root\n", encoding="utf-8")
            (repo / "agents" / "note.md").write_text("干净归档\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md", "agents/note.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c2 mask"], check=True)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = crh.main(["--repo", tmp, "--commit-range", f"{base_rev}..HEAD"])
            self.assertEqual(rc, 1)

    def test_invalid_commit_range_fails_and_valid_empty_range_passes(self):
        """回归 1：无效 commit 范围（如 missing_ref..HEAD）必须报错退出（返回码 1），合法空范围（HEAD..HEAD）返回 0。"""
        with tempfile.TemporaryDirectory() as tmp:
            init_temp_git_repo(tmp)
            receipt_bad = Path(tmp) / "bad_receipt.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_bad = crh.main([
                    "--repo", tmp,
                    "--commit-range", "missing_ref..HEAD",
                    "--json-receipt", str(receipt_bad),
                ])
            self.assertEqual(rc_bad, 1)
            data_bad = json.loads(receipt_bad.read_text(encoding="utf-8"))
            self.assertFalse(data_bad["passed"])
            self.assertTrue(len(data_bad["git_errors"]) >= 1)

            receipt_empty = Path(tmp) / "empty_receipt.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc_ok = crh.main([
                    "--repo", tmp,
                    "--commit-range", "HEAD..HEAD",
                    "--json-receipt", str(receipt_empty),
                ])
            self.assertEqual(rc_ok, 0)
            data_ok = json.loads(receipt_empty.read_text(encoding="utf-8"))
            self.assertTrue(data_ok["passed"])
            self.assertEqual(data_ok["scanned_commits"], [])

    def test_staged_index_scanned_even_when_worktree_cleaned(self):
        """回归 2：暂存区（git add）含敏感内容、随后工作区文件改回干净时，--include-worktree 必须扫描 index blob 并报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            (repo / "README.md").write_text("联系：staged_secret" + "@" + "private.org\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md"], check=True)
            (repo / "README.md").write_text("# Clean Root\n", encoding="utf-8")

            receipt_path = Path(tmp) / "staged_receipt.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = crh.main([
                    "--repo", tmp,
                    "--include-worktree",
                    "--json-receipt", str(receipt_path),
                ])
            self.assertEqual(rc, 1)
            data = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertIn("index:README.md", data["scanned_files"])
            self.assertEqual(len(data["privacy_errors"]), 1)

    def test_cross_line_deletion_and_rename_caught_by_ownership(self):
        """回归 3：修改 SemEval 文件同时删除学习线文件，或跨工作线重命名文件，ownership-mode error 必须拦截。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            learn_file = repo / "ML_note" / "原理笔记" / "BERT" / "note.md"
            semeval_file = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "exp.md"
            learn_file.parent.mkdir(parents=True, exist_ok=True)
            semeval_file.parent.mkdir(parents=True, exist_ok=True)
            learn_file.write_text("# Learn Note\n", encoding="utf-8")
            semeval_file.write_text("# SemEval V1\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "seed files"], check=True)
            seed_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            semeval_file.write_text("# SemEval V2\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "rm", "-q", str(learn_file.relative_to(repo))], check=True)
            subprocess.run(["git", "-C", tmp, "add", str(semeval_file.relative_to(repo))], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "edit semeval and delete learning"], check=True)

            with contextlib.redirect_stdout(io.StringIO()):
                rc_del = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{seed_rev}..HEAD",
                    "--ownership-mode", "error",
                ])
            self.assertEqual(rc_del, 1)

            before_rename = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
            renamed_target = repo / "ML_note" / "原理笔记" / "BERT" / "moved_exp.md"
            renamed_target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "git", "-C", tmp, "mv",
                str(semeval_file.relative_to(repo)),
                str(renamed_target.relative_to(repo)),
            ], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "cross-line rename"], check=True)

            with contextlib.redirect_stdout(io.StringIO()):
                rc_ren = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{before_rename}..HEAD",
                    "--ownership-mode", "error",
                ])
            self.assertEqual(rc_ren, 1)

    def test_redaction_never_echoes_secret_in_findings_stdout_or_json_receipt(self):
        """回归 4：命中模拟令牌、私人邮箱或容器路径时，返回值、终端输出与 JSON 回执均不得包含原始敏感值。"""
        fake_token = "ghp_" + ("A" * 32)
        fake_email = "leak_user_99" + "@" + "personal-domain.com"
        fake_path = "/" + "home/antigravity/secret_dir/key.pem"
        sample_content = f"token={fake_token}\nmail={fake_email}\npath={fake_path}\n"

        findings = crh.scan_text("README.md", sample_content)
        self.assertEqual(len(findings), 3)
        joined_findings = "\n".join(findings)
        self.assertNotIn(fake_token, joined_findings)
        self.assertNotIn(fake_email, joined_findings)
        self.assertNotIn(fake_path, joined_findings)

        with tempfile.TemporaryDirectory() as tmp:
            sample_file = Path(tmp) / "README.md"
            receipt_file = Path(tmp) / "receipt.json"
            sample_file.write_text(sample_content, encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = crh.main([
                    "--repo", tmp,
                    "--files", str(sample_file),
                    "--json-receipt", str(receipt_file),
                ])
            self.assertEqual(rc, 1)
            stdout_text = buf.getvalue()
            receipt_text = receipt_file.read_text(encoding="utf-8")
            for secret in (fake_token, fake_email, fake_path):
                self.assertNotIn(secret, stdout_text)
                self.assertNotIn(secret, receipt_text)

    def test_task_scope_catches_single_line_out_of_scope_edit(self):
        """回归 5（Codex 建议 1）：即使单次 commit 只改了一条线（未混线），只要越出当前任务允许范围即报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            base_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            # B1 任务只允许修改 SemEval 的 code/ 和 README.md，但提交者只改了学习笔记（单线、未混线）
            learn_note = repo / "ML_note" / "原理笔记" / "BERT" / "attention.md"
            learn_note.parent.mkdir(parents=True, exist_ok=True)
            learn_note.write_text("# Attention Clean Note\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "only edit learning note"], check=True)

            b1_prefixes = [
                "ML_note/实验/SemEval_DiCo-NLI/code/",
                "ML_note/实验/SemEval_DiCo-NLI/README.md",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                rc_wrong_line = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{base_rev}..HEAD",
                    "--expected-line", "semeval",
                    "--allowed-prefixes", *b1_prefixes,
                ])
            self.assertEqual(rc_wrong_line, 1)

            # 进一步验证：父目录前缀 ML_note/实验/SemEval_DiCo-NLI/ 不会误放行独立的 论文研究/ 子线
            issues_paper, _ = crh.check_ownership(
                ["ML_note/实验/SemEval_DiCo-NLI/论文研究/survey.md"],
                allowed_prefixes=["ML_note/实验/SemEval_DiCo-NLI/"],
            )
            self.assertEqual(len(issues_paper), 1)

            # 合规修改：修改 B1 允许的 code/ 文件 + 共用索引 README.md 应通过
            issues_ok, _ = crh.check_ownership(
                ["ML_note/实验/SemEval_DiCo-NLI/code/b1_infer.py", "README.md"],
                expected_line="semeval",
                allowed_prefixes=b1_prefixes,
            )
            self.assertEqual(issues_ok, [])

    def test_task_manifest_and_handoff_header_validation(self):
        """回归 6（Codex 建议 3）：解析并校验交接文件的 6 项必要元数据及可见范围门禁。"""
        valid_md = (
            "# ShortGPT BI 删层原理笔记\n\n"
            "| 字段 | 内容 |\n"
            "|---|---|\n"
            "| 任务与当前负责人 | `TASK-LEARN-001 | Gemini 网页端` |\n"
            "| 允许修改范围 | `ML_note/原理笔记/`、`ML_note/实验/ShortGPT_BI删层/` |\n"
            "| 来源版本 | `commit:72bf9765 | drive:1DLCgS6Z` |\n"
            "| 可见范围 | `工作区私有` |\n"
            "| 当前状态 | `待审` |\n"
            "| 正式结果链接 | `ML_note/原理笔记/推理优化/ShortGPT_BI删层笔记.md` |\n"
            "| 目标仓库路径 | `ML_note/原理笔记/推理优化/ShortGPT_BI删层笔记.md` |\n\n"
            "正文内容干净，公式 $s_i = 1 - \\cos(X_i, X_{i+1})$。\n"
        )
        m = crh.parse_task_manifest_text(valid_md)
        self.assertEqual(m["task_id"], "TASK-LEARN-001")
        self.assertEqual(m["owner"], "Gemini 网页端")
        self.assertEqual(m["allowed_prefixes"], ["ML_note/原理笔记/", "ML_note/实验/ShortGPT_BI删层/"])
        self.assertEqual(crh.validate_handoff_manifest(m, require_public_ready=False), [])
        # 未标为“允许公开发布”时，--require-public-ready 必须拦截
        pub_errs = crh.validate_handoff_manifest(m, require_public_ready=True)
        self.assertEqual(len(pub_errs), 1)

    def test_version_bound_receipt_invalidated_when_file_modified_after_check(self):
        """回归 7（Drive 同步规则 3）：回执绑定文件 SHA-256；若检查后文件被再次修改，旧回执自动失效。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            draft = Path(tmp) / "TASK-LEARN-001_draft.md"
            receipt_json = Path(tmp) / "receipt.json"
            receipt_md = Path(tmp) / "receipt.md"

            content_v1 = (
                "# BI 删层笔记草稿\n\n"
                "| 字段 | 内容 |\n"
                "|---|---|\n"
                "| 任务与当前负责人 | `TASK-LEARN-001 | Gemini 网页端` |\n"
                "| 允许修改范围 | `ML_note/原理笔记/` |\n"
                "| 来源版本 | `commit:72bf9765` |\n"
                "| 可见范围 | `工作区私有` |\n"
                "| 当前状态 | `待审` |\n"
                "| 正式结果链接 | `ML_note/原理笔记/推理优化/BI删层.md` |\n"
                "| 目标仓库路径 | `ML_note/原理笔记/推理优化/BI删层.md` |\n\n"
                "版本 1：无离群值时量化步长 $\\Delta = 1.0 / 127$。\n"
            )
            draft.write_text(content_v1, encoding="utf-8")

            # 第 1 步：对 v1 运行检查并签发 JSON + Markdown 回执
            with contextlib.redirect_stdout(io.StringIO()):
                rc_v1 = crh.main([
                    "--repo", str(repo),
                    "--files", str(draft),
                    "--require-handoff-header",
                    "--drive-file-id", "drive_test_file_001",
                    "--json-receipt", str(receipt_json),
                    "--markdown-receipt", str(receipt_md),
                ])
            self.assertEqual(rc_v1, 0)
            self.assertTrue(receipt_md.is_file())

            # 第 2 步：文件未改动时，--verify-receipt 校验通过
            with contextlib.redirect_stdout(io.StringIO()):
                rc_verify_ok = crh.main([
                    "--repo", str(repo),
                    "--files", str(draft),
                    "--require-handoff-header",
                    "--verify-receipt", str(receipt_json),
                ])
            self.assertEqual(rc_verify_ok, 0)

            # 第 3 步：模拟 Gemini 在检查期间又改了 Drive 文件（v2），旧回执必须失效（返回码 1）
            draft.write_text(content_v1 + "\n新增未送检段落。\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_verify_stale = crh.main([
                    "--repo", str(repo),
                    "--files", str(draft),
                    "--require-handoff-header",
                    "--verify-receipt", str(receipt_json),
                ])
            self.assertEqual(rc_verify_stale, 1)
            self.assertIn("版本失效", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
