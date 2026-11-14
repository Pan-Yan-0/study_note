#!/usr/bin/env python3
"""agents/check_repo_hygiene.py 的回归测试套件（含 v2 边界回归、v3 任务范围/回执回归与 v4 三项门禁闭环回归）。"""
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

            issues_paper, _ = crh.check_ownership(
                ["ML_note/实验/SemEval_DiCo-NLI/论文研究/survey.md"],
                allowed_prefixes=["ML_note/实验/SemEval_DiCo-NLI/"],
            )
            self.assertEqual(len(issues_paper), 1)

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
        pub_errs = crh.validate_handoff_manifest(m, require_public_ready=True)
        self.assertTrue(len(pub_errs) >= 1)

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

            with contextlib.redirect_stdout(io.StringIO()):
                rc_verify_ok = crh.main([
                    "--repo", str(repo),
                    "--files", str(draft),
                    "--require-handoff-header",
                    "--verify-receipt", str(receipt_json),
                ])
            self.assertEqual(rc_verify_ok, 0)

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

    def test_public_ready_gate_in_commit_and_worktree_scan(self):
        """回归 8（Codex v4 问题 1）：在 commit 扫描或工作区扫描中传入 --require-public-ready 时，标记为“工作区私有”或“待审”的文件必须报错拦截。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            base_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            note_file = repo / "ML_note" / "原理笔记" / "推理优化" / "draft.md"
            note_file.parent.mkdir(parents=True, exist_ok=True)
            note_file.write_text(
                "# 草稿\n\n| 字段 | 内容 |\n|---|---|\n| 可见范围 | `工作区私有` |\n| 当前状态 | `待审` |\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "commit private draft"], check=True)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_commit = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{base_rev}..HEAD",
                    "--require-public-ready",
                ])
            self.assertEqual(rc_commit, 1)
            self.assertIn("公开发布拦截", buf.getvalue())

            # 工作区未提交改动同样拦截
            note_file.write_text("# 草稿\n可见范围: 工作区私有\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                rc_wt = crh.main([
                    "--repo", tmp,
                    "--include-worktree",
                    "--require-public-ready",
                ])
            self.assertEqual(rc_wt, 1)

            # 改为“允许公开发布”且“通过”后，新 commit 范围通过
            prev_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
            note_file.write_text(
                "# 正式笔记\n\n| 字段 | 内容 |\n|---|---|\n| 可见范围 | `允许公开发布` |\n| 当前状态 | `通过` |\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "publish ready note"], check=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rc_ok = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{prev_rev}..HEAD",
                    "--require-public-ready",
                ])
            self.assertEqual(rc_ok, 0)

    def test_require_task_scope_reads_repo_binding_and_fails_when_missing(self):
        """回归 9（Codex v4 问题 2）：--require-task-scope 在父提交缺失任务绑定时报错；单独提交 agents/current_task.json 后按父提交绑定拦截越界修改。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            base_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            learn_file = repo / "ML_note" / "原理笔记" / "BERT" / "only_learn.md"
            learn_file.parent.mkdir(parents=True, exist_ok=True)
            learn_file.write_text("# Clean Learn\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c1 no task binding"], check=True)
            c1_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            # 场景 A：既未传 CLI 范围，父提交也无 agents/current_task.json -> 必须报“未完成任务范围验收”
            buf_missing = io.StringIO()
            with contextlib.redirect_stdout(buf_missing):
                rc_missing = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{base_rev}..HEAD",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_missing, 1)
            self.assertIn("未完成任务范围验收", buf_missing.getvalue())

            # 单独提交 agents/current_task.json（仅激活 TASK-SEMEVAL-B1）
            (repo / "agents").mkdir(exist_ok=True)
            (repo / "agents" / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-SEMEVAL-B1",
                            "owner": "Antigravity (SemEval)",
                            "expected_line": "semeval",
                            "allowed_prefixes": [
                                "ML_note/实验/SemEval_DiCo-NLI/code/",
                                "ML_note/实验/SemEval_DiCo-NLI/README.md",
                            ],
                        }
                    ],
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c_bind: standalone current_task.json"], check=True)
            c_bind_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            # 场景 B：父提交配置了 B1 任务范围，不传手动 CLI 参数扫描只改学习线的 commit -> 必须报“越界修改”
            learn_file.write_text("# Clean Learn v2\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "ML_note/原理笔记/BERT/only_learn.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "c2 invalid learn change"], check=True)

            buf_b1 = io.StringIO()
            with contextlib.redirect_stdout(buf_b1):
                rc_b1_violation = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{c_bind_rev}..HEAD",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_b1_violation, 1)
            self.assertIn("越界修改", buf_b1.getvalue())

            # 场景 C：在 B1 任务绑定下修改允许的 SemEval code/ 文件 -> 不传手动 CLI 参数亦可通过
            prev_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
            b1_code = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "code" / "b1.py"
            b1_code.parent.mkdir(parents=True, exist_ok=True)
            b1_code.write_text("# B1 inference\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "b1 valid change"], check=True)

            with contextlib.redirect_stdout(io.StringIO()):
                rc_b1_ok = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{prev_rev}..HEAD",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_b1_ok, 0)

    def test_verify_bound_receipt_rejects_empty_input_and_populates_commit_hashes(self):
        """回归 10（Codex v4 问题 3）：verify_bound_receipt 拒绝空输入 {}；commit 扫描正常填充 file_hashes。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            base_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
            (repo / "README.md").write_text("# Updated Root\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "README.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "update readme"], check=True)

            receipt_json = Path(tmp) / "commit_receipt.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{base_rev}..HEAD",
                    "--json-receipt", str(receipt_json),
                ])
            self.assertEqual(rc, 0)
            data = json.loads(receipt_json.read_text(encoding="utf-8"))
            self.assertIn("README.md", data["file_hashes"])

            # 直接调用 verify_bound_receipt(receipt, {}) 必须报错拒绝空输入
            empty_errs = crh.verify_bound_receipt(receipt_json, {})
            self.assertEqual(len(empty_errs), 1)
            self.assertIn("为空", empty_errs[0])

            # CLI 传入 --verify-receipt 但未传 --files 时必须报错退出
            with contextlib.redirect_stdout(io.StringIO()):
                rc_no_files = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{base_rev}..HEAD",
                    "--verify-receipt", str(receipt_json),
                ])
            self.assertEqual(rc_no_files, 1)

    def test_first_push_zero_remote_sha_scans_full_history_and_catches_reverted_private_draft(self):
        """回归 11（Codex v5 首次推送漏扫历史）：remote_sha 为全零时扫描全部将被发布的历史，拦截中间 commit 写入后又删除的私有草稿。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            # Commit 1a：单独提交任务绑定清单 agents/current_task.json
            agents_dir = repo / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / "current_task.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "active_tasks": [
                            {
                                "task_id": "TASK-INFRA-005",
                                "owner": "基建线",
                                "expected_line": "infra",
                                "allowed_prefixes": ["agents/"],
                                "allow_shared_index": True,
                            },
                            {
                                "task_id": "TASK-LEARN-001",
                                "owner": "学习线",
                                "expected_line": "learning",
                                "allowed_prefixes": ["ML_note/原理笔记/"],
                                "allow_shared_index": True,
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "commit 1a: standalone task binding"], check=True)

            # Commit 1b：提交检查脚本
            (agents_dir / "check_repo_hygiene.py").write_bytes(Path(crh.__file__).read_bytes())
            subprocess.run(["git", "-C", tmp, "add", "agents/check_repo_hygiene.py"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "commit 1b: add checker script"], check=True)

            # Commit 2：加入标记“工作区私有”的 NLP 草稿
            draft = repo / "ML_note" / "原理笔记" / "BERT" / "private_draft.md"
            draft.parent.mkdir(parents=True, exist_ok=True)
            draft.write_text(
                "# 私有草稿\n\n| 字段 | 内容 |\n|---|---|\n| 可见范围 | `工作区私有` |\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "commit 2: add private draft"], check=True)

            # Commit 3：删除该私有草稿（但仍在 Git 历史中）
            subprocess.run(["git", "-C", tmp, "rm", "-q", "ML_note/原理笔记/BERT/private_draft.md"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "commit 3: delete private draft"], check=True)
            head_sha = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
            zero_sha = "0" * 40

            # 1) 验证 --commit-range 0000...000..HEAD 扫描全部历史并拦截 Commit 2 的私有草稿
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_cli = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{zero_sha}..{head_sha}",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                    "--require-public-ready",
                ])
            self.assertEqual(rc_cli, 1)
            self.assertIn("公开发布拦截", buf.getvalue())

            # 2) 验证实际执行 pre-push 钩子（stdin 传入 remote_sha=0000...000）同样拦截
            hook_path = crh.install_pre_push_hook(repo)
            stdin_line = f"refs/heads/note {head_sha} refs/heads/note {zero_sha}\n"
            proc = subprocess.run(
                [str(hook_path)],
                cwd=tmp,
                input=stdin_line,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("公开发布拦截", proc.stdout + proc.stderr)

    def test_active_tasks_list_allows_multiple_worklines_and_enforces_forbidden_prefixes(self):
        """回归 12（C-006 问题 #1 与 #5）：active_tasks 支持多工作线并行提交，且 forbidden_prefixes 精确拦截禁止子目录（如 code/a1_train/）。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            agents_dir = repo / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            # 先模拟旧状态：绑定单任务 TASK-INFRA-004
            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "task_id": "TASK-INFRA-004",
                    "owner": "Antigravity (基建对话)",
                    "expected_line": "infra",
                    "allowed_prefixes": ["AGENTS.md", "agents/", ".github/"],
                    "status": "通过",
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "old infra binding"], check=True)
            old_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            # 1) 单独提交升级为 active_tasks（含 TASK-INFRA-005、TASK-SEMEVAL-A1-RECALC、TASK-SEMEVAL-B1、TASK-LEARN-001）
            (agents_dir / "current_task.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "active_tasks": [
                            {
                                "task_id": "TASK-INFRA-005",
                                "owner": "Antigravity (基建对话)",
                                "expected_line": "infra",
                                "allowed_prefixes": ["AGENTS.md", "agents/", ".github/"],
                            },
                            {
                                "task_id": "TASK-SEMEVAL-A1-RECALC",
                                "owner": "Antigravity (SemEval 对话)",
                                "expected_line": "semeval",
                                "allowed_prefixes": [
                                    "ML_note/实验/SemEval_DiCo-NLI/code/",
                                    "ML_note/实验/SemEval_DiCo-NLI/metrics/a1_train/formal_v1/",
                                    "ML_note/实验/SemEval_DiCo-NLI/实验状态与算力计划.md",
                                    "ML_note/实验/SemEval_DiCo-NLI/README.md",
                                    "ML_note/实验/SemEval_DiCo-NLI/实验登记表.md",
                                ],
                                "forbidden_prefixes": [
                                    "ML_note/实验/SemEval_DiCo-NLI/code/a1_train/",
                                ],
                            },
                            {
                                "task_id": "TASK-LEARN-001",
                                "owner": "Gemini 网页端",
                                "expected_line": "learning",
                                "allowed_prefixes": [
                                    "ML_note/原理笔记/",
                                    "ML_note/实验/ShortGPT_BI删层/",
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "upgrade to active_tasks"], check=True)

            # 2) SemEval 线正常提交（改 实验状态与算力计划.md + 新复算脚本 code/recalc_a1.py）-> 放行
            semeval_plan = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "实验状态与算力计划.md"
            semeval_recalc = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "code" / "recalc_a1.py"
            semeval_plan.parent.mkdir(parents=True, exist_ok=True)
            semeval_recalc.parent.mkdir(parents=True, exist_ok=True)
            semeval_plan.write_text("# 状态更新\n", encoding="utf-8")
            semeval_recalc.write_text("# recalc script\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "semeval recalc commit"], check=True)

            # 3) 学习线正常提交（改 ML_note/原理笔记/BERT/x.md）-> 放行
            learn_note = repo / "ML_note" / "原理笔记" / "BERT" / "x.md"
            learn_note.parent.mkdir(parents=True, exist_ok=True)
            learn_note.write_text("# BERT Note\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "learning note commit"], check=True)

            with contextlib.redirect_stdout(io.StringIO()):
                rc_parallel = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{old_rev}..HEAD",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_parallel, 0)

            # 4) 触碰 forbidden_prefixes（ML_note/实验/SemEval_DiCo-NLI/code/a1_train/run_a1.py）-> 必须报错拦截
            good_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()
            a1_code = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "code" / "a1_train" / "run_a1.py"
            a1_code.parent.mkdir(parents=True, exist_ok=True)
            a1_code.write_text("# forbidden modification\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "touch forbidden a1_train"], check=True)

            buf_forb = io.StringIO()
            with contextlib.redirect_stdout(buf_forb):
                rc_forb = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{good_rev}..HEAD",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_forb, 1)
            self.assertIn("禁止修改范围", buf_forb.getvalue())

            # 5) 触碰未激活的工作线（ML_note/实验/SemEval_DiCo-NLI/论文研究/p.md）-> 必须报错拦截
            subprocess.run(["git", "-C", tmp, "reset", "--hard", "-q", good_rev], check=True)
            paper_file = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "论文研究" / "p.md"
            paper_file.parent.mkdir(parents=True, exist_ok=True)
            paper_file.write_text("# Paper note\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "touch inactive semeval_papers"], check=True)

            buf_inact = io.StringIO()
            with contextlib.redirect_stdout(buf_inact):
                rc_inact = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{good_rev}..HEAD",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_inact, 1)
            self.assertIn("越界修改", buf_inact.getvalue())

    def test_parent_commit_binding_prevents_self_escalation_and_mixed_task_binding_edit(self):
        """回归 13（C-006 问题 #4）：强制 agents/current_task.json 单独提交，且按父提交 C^ 校验，拦截同 commit 自改绑定越界。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            agents_dir = repo / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-INFRA-004",
                            "owner": "Antigravity (基建对话)",
                            "expected_line": "infra",
                            "allowed_prefixes": ["agents/"],
                        }
                    ],
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "bind infra only"], check=True)
            infra_rev = subprocess.check_output(["git", "-C", tmp, "rev-parse", "HEAD"], text=True).strip()

            # 漏洞重现 1：同一个 commit 把 current_task.json 改成 SemEval 线并修改 SemEval 文件 -> 必须拦截！
            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-SEMEVAL-B1",
                            "owner": "Antigravity (SemEval)",
                            "expected_line": "semeval",
                            "allowed_prefixes": ["ML_note/实验/SemEval_DiCo-NLI/"],
                        }
                    ],
                }),
                encoding="utf-8",
            )
            semeval_file = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "实验状态与算力计划.md"
            semeval_file.parent.mkdir(parents=True, exist_ok=True)
            semeval_file.write_text("# Hijacked\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "self escalation attempt"], check=True)

            buf_esc = io.StringIO()
            with contextlib.redirect_stdout(buf_esc):
                rc_esc = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{infra_rev}..HEAD",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_esc, 1)
            self.assertIn("必须单独提交", buf_esc.getvalue())

            # 漏洞重现 2：即使同属 infra 线，把 current_task.json 与其他 infra 文件混在同一个 commit 提交 -> 同样拦截！
            subprocess.run(["git", "-C", tmp, "reset", "--hard", "-q", infra_rev], check=True)
            (agents_dir / "纠错台账.md").write_text("# 台账\n", encoding="utf-8")
            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-INFRA-005",
                            "owner": "Antigravity (基建对话)",
                            "expected_line": "infra",
                            "allowed_prefixes": ["agents/"],
                        }
                    ],
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "mixed infra commit"], check=True)

            buf_mix = io.StringIO()
            with contextlib.redirect_stdout(buf_mix):
                rc_mix = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{infra_rev}..HEAD",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_mix, 1)
            self.assertIn("必须单独提交", buf_mix.getvalue())

            # 漏洞重现 3：单独提交 current_task.json 但包含跨工作线非法前缀 -> 严格 schema 校验拦截！
            subprocess.run(["git", "-C", tmp, "reset", "--hard", "-q", infra_rev], check=True)
            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-SEMEVAL-B1",
                            "owner": "Antigravity (SemEval)",
                            "expected_line": "semeval",
                            "allowed_prefixes": ["ML_note/原理笔记/"],
                        }
                    ],
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "invalid schema binding"], check=True)

            buf_schema = io.StringIO()
            with contextlib.redirect_stdout(buf_schema):
                rc_schema = crh.main([
                    "--repo", tmp,
                    "--commit-range", f"{infra_rev}..HEAD",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_schema, 1)
            self.assertIn("跨工作线路径前缀", buf_schema.getvalue())

            # 漏洞重现 4：未提交改动同时修改 current_task.json 与业务文件 -> --include-worktree 下同样拦截！
            subprocess.run(["git", "-C", tmp, "reset", "--hard", "-q", infra_rev], check=True)
            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-SEMEVAL-B1",
                            "owner": "Antigravity (SemEval)",
                            "expected_line": "semeval",
                            "allowed_prefixes": ["ML_note/实验/SemEval_DiCo-NLI/"],
                        }
                    ],
                }),
                encoding="utf-8",
            )
            semeval_file.parent.mkdir(parents=True, exist_ok=True)
            semeval_file.write_text("# Uncommitted mixed\n", encoding="utf-8")

            buf_wt = io.StringIO()
            with contextlib.redirect_stdout(buf_wt):
                rc_wt = crh.main([
                    "--repo", tmp,
                    "--include-worktree",
                    "--ownership-mode", "error",
                    "--require-task-scope",
                ])
            self.assertEqual(rc_wt, 1)
            self.assertIn("必须单独提交", buf_wt.getvalue())

    def test_task_pool_check_catches_false_synced_claim_and_validates_pools(self):
        """回归 14（C-006 问题 #2 任务级分端信息共享池校验）：验证 agents/task_pools/*.json 结构及拦截只交 README 漏交数据的假同步。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = init_temp_git_repo(tmp)
            agents_dir = repo / "agents"
            pools_dir = agents_dir / "task_pools"
            pools_dir.mkdir(parents=True, exist_ok=True)

            # 创建 formal_v1/README.md 并入库，但故意不提交 summary.json（模拟 .gitignore 漏 git add -f）
            formal_readme = repo / "ML_note" / "实验" / "SemEval_DiCo-NLI" / "metrics" / "a1_train" / "formal_v1" / "README.md"
            formal_readme.parent.mkdir(parents=True, exist_ok=True)
            formal_readme.write_text("# A1 Formal Summary\n", encoding="utf-8")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "add formal_v1 readme only"], check=True)

            (agents_dir / "current_task.json").write_text(
                json.dumps({
                    "schema_version": "2.0",
                    "active_tasks": [
                        {
                            "task_id": "TASK-INFRA-005",
                            "owner": "Antigravity (基建对话)",
                            "expected_line": "infra",
                            "allowed_prefixes": ["agents/"],
                        },
                        {
                            "task_id": "TASK-SEMEVAL-A1-RECALC",
                            "owner": "Antigravity (SemEval 对话)",
                            "expected_line": "semeval",
                            "allowed_prefixes": ["ML_note/实验/SemEval_DiCo-NLI/"],
                            "pool_path": "agents/task_pools/TASK-SEMEVAL-A1-RECALC.json",
                        },
                    ],
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", tmp, "add", "agents/current_task.json"], check=True)
            subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "bind task with pool_path"], check=True)

            # 1) 真实标记 github_repo="partial_readme_only" 且声明 blocked_on_unshared_artifact -> 校验通过
            pool_file = pools_dir / "TASK-SEMEVAL-A1-RECALC.json"
            pool_data = {
                "schema_version": "1.0",
                "task_id": "TASK-SEMEVAL-A1-RECALC",
                "expected_line": "semeval",
                "owner": "Antigravity (SemEval 对话)",
                "endpoints": {
                    "codex": {"role": "archiver", "state": "verified_in_sandbox_partial_pushed"},
                    "antigravity_semeval": {
                        "role": "independent_recalculator",
                        "state": "blocked_on_unshared_artifact",
                        "blocked_by": ["ART-A1-RESULTS"],
                    },
                },
                "artifacts": [
                    {
                        "artifact_id": "ART-A1-RESULTS",
                        "name": "a1_train_v1_results.zip",
                        "holders": {
                            "codex": "held_unpushed",
                            "antigravity_semeval": "missing",
                            "github_repo": "partial_readme_only",
                        },
                        "repo_path": "ML_note/实验/SemEval_DiCo-NLI/metrics/a1_train/formal_v1/",
                        "required_repo_files_for_recalc": [
                            "ML_note/实验/SemEval_DiCo-NLI/metrics/a1_train/formal_v1/README.md",
                            "ML_note/实验/SemEval_DiCo-NLI/metrics/a1_train/formal_v1/summary.json",
                        ],
                    }
                ],
            }
            pool_file.write_text(json.dumps(pool_data, ensure_ascii=False, indent=2), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                rc_honest = crh.main([
                    "--repo", tmp,
                    "--check-task-pool",
                ])
            self.assertEqual(rc_honest, 0)

            # 2) 若谎报 github_repo="synced" 而 summary.json 仍未入库 -> 必须报错“任务信息共享池假同步拦截”
            pool_data["artifacts"][0]["holders"]["github_repo"] = "synced"
            pool_file.write_text(json.dumps(pool_data, ensure_ascii=False, indent=2), encoding="utf-8")

            buf_false_sync = io.StringIO()
            with contextlib.redirect_stdout(buf_false_sync):
                rc_false_sync = crh.main([
                    "--repo", tmp,
                    "--check-task-pool",
                ])
            self.assertEqual(rc_false_sync, 1)
            self.assertIn("任务信息共享池假同步拦截", buf_false_sync.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)


