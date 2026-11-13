#!/usr/bin/env python3
"""仓库路径归属与隐私/本机路径自动检查脚本。

满足四项核心要求：
1. 路径归属检查：单次提交只改同一工作线目录；共用索引（根 README.md、ML_note/实验结果汇总.md）允许随同更新；
   ML_note/实验/SemEval_DiCo-NLI/论文研究/ 按独立工作线判定。支持 --ownership-mode warn|error（默认 warn）。
2. 上下文感知的隐私与本机路径扫描：拦截私有学习存档字段、个人邮箱、密钥令牌与本地容器路径前缀，
   同时允许 NLP 笔记中的正常缩写（如 Lv et al.）。
3. 白名单精确到具体匹配值（公开 noreply 邮箱、平台路径 /kaggle/working 等），不按整文件豁免；
   共用索引文件与 agents/ 归档照常扫描。
4. 逐个 commit 扫描（覆盖一次 push 范围内的每个 commit，防止前序 commit 混入隐私后被后续 commit 掩盖）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SHARED_INDEX_FILES = {"README.md", "ML_note/实验结果汇总.md"}

ALLOWED_EMAILS = {
    "noreply@anthropic.com",
    "noreply@openai.com",
    "112330427+Pan-Yan-0@users.noreply.github.com",
}

ALLOWED_PATH_PREFIXES = (
    "/kaggle/working",
    "/kaggle/input",
    "/mnt/workspace",
    "/content",
    "/tmp",
    "/path/to/",
    "/absolute/path/to/",
    "/var/colab/",
)

# 1. 明确的私有存档字段（拆字拼接，使脚本自身无需整文件豁免即可通过全文扫描）
_ARCHIVE_TOKENS = [
    "经" + "验值",
    "累计" + "经验",
    "经" + r"验\s*[+＋]\s*\d",
    "签" + "到",
    "学习" + "画像",
    "等" + r"级\s*\d",
    "连" + r"续\s?\d+\s?天",
    "技能" + "树",
    "徽" + "章",
    "称" + "号",
    "复习" + "队列",
    "档" + r"位\s*\d",
    "\u2b50",
    r"[✓△✗]\s*[+＋]\s*\d+",
    r"(?<![A-Za-z0-9_])" + "X" + r"P(?![A-Za-z0-9_])",
]
PRIVATE_ARCHIVE_RE = re.compile("|".join(_ARCHIVE_TOKENS))

# 2. 结合私有存档字段与上下文判定的等级缩写 / 关卡字段（防 NLP 正常缩写误报）
_LV_CTX = "(?:经" + "验|等" + "级|升" + "级|存" + "档|称" + "号|签" + "到)"
CONTEXT_LV_RE = re.compile(
    r"(?<![A-Za-z0-9_])L" + r"v\s?\d+"
    r"|" + _LV_CTX + r"[^\n]{0,20}(?<![A-Za-z0-9_])L" + r"v(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_])L" + r"v(?![A-Za-z0-9_])[^\n]{0,20}" + _LV_CTX
)

_BOSS_CTX = "(?:格" + "血|扣" + "血|掉" + "血|击" + "败|第" + r"\s*12\s*节|存" + "档)"
CONTEXT_BOSS_RE = re.compile(
    _BOSS_CTX + r"[^\n]{0,25}(?<![A-Za-z0-9_])Bo" + r"ss(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_])Bo" + r"ss(?![A-Za-z0-9_])[^\n]{0,25}" + _BOSS_CTX
)

# 3. 邮箱、密钥令牌与本机/容器路径
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
TOKEN_RE = re.compile(
    r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,})\b"
)
LOCAL_PATH_RE = re.compile(
    r"(?:file:///" + r"home/[^\s)\]`\"'>]+|(?<![A-Za-z0-9_])/" + r"home/[A-Za-z0-9_.-]+(?:/[^\s)\]`\"'>]*)?"
    r"|(?<![A-Za-z0-9_])/" + r"Users/[A-Za-z0-9_.-]+(?:/[^\s)\]`\"'>]*)?|[A-Za-z]:\\Users\\[^\s)\]`\"'>]+)"
)


def classify_line(rel_path: str) -> str:
    """将仓库相对路径映射为工作线名称。"""
    p = rel_path.replace("\\", "/").lstrip("./")
    if p in SHARED_INDEX_FILES:
        return "shared_index"
    if p.startswith("ML_note/实验/SemEval_DiCo-NLI/论文研究/"):
        return "semeval_papers"
    if p.startswith("ML_note/实验/SemEval_DiCo-NLI/"):
        return "semeval"
    if p.startswith(("ML_note/原理笔记/", "ML_note/论文阅读/", "ML_note/实验/ShortGPT_BI删层/", "ML_note/吴恩达机器学习/")):
        return "learning"
    if p in {"AGENTS.md", "CLAUDE.md", ".gitignore"} or p.startswith("agents/"):
        return "infra"
    return "other"


def check_ownership(paths: Iterable[str]) -> tuple[list[str], dict[str, list[str]]]:
    """检查一组改动路径是否跨越多个非共用工作线。"""
    by_line: dict[str, list[str]] = {}
    for p in sorted(set(paths)):
        if not p:
            continue
        line = classify_line(p)
        by_line.setdefault(line, []).append(p)
    non_shared = {k: v for k, v in by_line.items() if k != "shared_index"}
    issues: list[str] = []
    if len(non_shared) > 1:
        summary = ", ".join(f"{k}({len(v)}个文件: {v[0]}等)" for k, v in sorted(non_shared.items()))
        issues.append(f"单次提交跨越多个工作线：{summary}")
    return issues, by_line


def is_allowed_path(matched_path: str) -> bool:
    for prefix in ALLOWED_PATH_PREFIXES:
        if matched_path.startswith(prefix):
            return True
    return False


def scan_text(label: str, text: str) -> list[str]:
    """逐行扫描单个文本内容，返回违规项列表（无整文件豁免）。"""
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for rx, kind in (
            (PRIVATE_ARCHIVE_RE, "私有存档字段"),
            (CONTEXT_LV_RE, "私有等级字段"),
            (CONTEXT_BOSS_RE, "私有Boss字段"),
            (TOKEN_RE, "密钥令牌"),
        ):
            for m in rx.finditer(line):
                findings.append(f"{label}:{lineno} 命中{kind}：{m.group(0)}")

        for m in EMAIL_RE.finditer(line):
            val = m.group(0)
            if val not in ALLOWED_EMAILS:
                findings.append(f"{label}:{lineno} 命中非白名单邮箱：{val}")

        for m in LOCAL_PATH_RE.finditer(line):
            val = m.group(0)
            if not is_allowed_path(val):
                findings.append(f"{label}:{lineno} 命中本地或容器路径：{val}")
    return findings


def git_run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def scan_commit(repo: Path, rev: str) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """扫描单个 commit 的路径归属、diff 新增行及该 commit 下改动文件的完整内容。"""
    r = git_run(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=ACMR", rev)
    changed = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    ownership_issues, by_line = check_ownership(changed)
    ownership_issues = [f"[{rev[:8]}] {msg}" for msg in ownership_issues]

    privacy_findings: list[str] = []
    for rel_path in changed:
        blob = git_run(repo, "show", f"{rev}:{rel_path}")
        if blob.returncode == 0 and "\x00" not in blob.stdout:
            privacy_findings.extend(scan_text(f"{rev[:8]}:{rel_path}", blob.stdout))
    return ownership_issues, privacy_findings, by_line


def default_commit_range(repo: Path) -> str | None:
    for base in ("origin/note", "@{u}"):
        if git_run(repo, "rev-parse", "--verify", "-q", base).returncode == 0:
            return f"{base}..HEAD"
    return None


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="Git 仓库根目录")
    ap.add_argument("--commit-range", help="要逐个扫描的 commit 范围，如 origin/note..HEAD")
    ap.add_argument("--files", nargs="*", help="直接扫描指定文件（相对仓库根目录或绝对路径）")
    ap.add_argument("--include-worktree", action="store_true", help="同时检查当前工作区未提交的改动")
    ap.add_argument(
        "--ownership-mode",
        choices=("warn", "error"),
        default="warn",
        help="路径归属跨线时的处理方式（默认 warn 试跑，验收或严格模式用 error）",
    )
    ap.add_argument("--json-receipt", help="输出 JSON 验收回执的路径")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    ownership_warnings: list[str] = []
    ownership_errors: list[str] = []
    privacy_errors: list[str] = []
    scanned_commits: list[str] = []
    scanned_files: list[str] = []

    if args.files is not None:
        rel_list: list[str] = []
        for f in args.files:
            fp = Path(f)
            if not fp.is_absolute():
                fp = repo / fp
            fp = fp.resolve()
            try:
                rel = str(fp.relative_to(repo))
            except ValueError:
                rel = fp.name
            rel_list.append(rel)
            scanned_files.append(rel)
            text = fp.read_text(encoding="utf-8", errors="replace")
            privacy_errors.extend(scan_text(rel, text))
        own_issues, _ = check_ownership(rel_list)
        if args.ownership_mode == "error":
            ownership_errors.extend(own_issues)
        else:
            ownership_warnings.extend(own_issues)
    else:
        rng = args.commit_range or default_commit_range(repo)
        if rng:
            revs_proc = git_run(repo, "rev-list", "--reverse", rng)
            revs = [l.strip() for l in revs_proc.stdout.splitlines() if l.strip()]
            for rev in revs:
                scanned_commits.append(rev)
                own_issues, priv_issues, _ = scan_commit(repo, rev)
                if args.ownership_mode == "error":
                    ownership_errors.extend(own_issues)
                else:
                    ownership_warnings.extend(own_issues)
                privacy_errors.extend(priv_issues)

        if args.include_worktree or not scanned_commits:
            wt = set(git_run(repo, "diff", "--name-only", "HEAD").stdout.splitlines())
            wt |= set(git_run(repo, "ls-files", "--others", "--exclude-standard").stdout.splitlines())
            wt_files = sorted(f for f in wt if f and (repo / f).is_file())
            own_issues, _ = check_ownership(wt_files)
            own_issues = [f"[worktree] {m}" for m in own_issues]
            if args.ownership_mode == "error":
                ownership_errors.extend(own_issues)
            else:
                ownership_warnings.extend(own_issues)
            for rel in wt_files:
                scanned_files.append(rel)
                raw = (repo / rel).read_bytes()
                if b"\x00" not in raw:
                    privacy_errors.extend(scan_text(rel, raw.decode("utf-8", errors="replace")))

    receipt = {
        "script": "agents/check_repo_hygiene.py",
        "script_sha256": script_sha256(),
        "ownership_mode": args.ownership_mode,
        "scanned_commits": scanned_commits,
        "scanned_files": scanned_files,
        "ownership_warnings": ownership_warnings,
        "ownership_errors": ownership_errors,
        "privacy_errors": privacy_errors,
        "passed": len(ownership_errors) == 0 and len(privacy_errors) == 0,
    }
    if args.json_receipt:
        out_path = Path(args.json_receipt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for w in ownership_warnings:
        print(f"提醒：{w}")
    for e in ownership_errors:
        print(f"错误（路径归属）：{e}")
    for e in privacy_errors:
        print(f"错误（隐私/路径）：{e}")

    total_err = len(ownership_errors) + len(privacy_errors)
    print(
        f"检查完成：扫描 {len(scanned_commits)} 个 commit、{len(scanned_files)} 个文件；"
        f"{total_err} 个错误，{len(ownership_warnings)} 个提醒。"
    )
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
