#!/usr/bin/env python3
"""仓库路径归属与隐私/本机路径自动检查脚本。

满足核心验收要求：
1. 路径归属检查：单次提交只改同一工作线目录；共用索引（根 README.md、ML_note/实验结果汇总.md）允许随同更新；
   ML_note/实验/SemEval_DiCo-NLI/论文研究/ 按独立工作线判定。归属检查覆盖新增、修改、删除（D）以及重命名（R）的源与目标路径。
   支持 --ownership-mode warn|error（默认 warn）。
2. 上下文感知的隐私与本机路径扫描：拦截私有学习字段、个人邮箱、密钥令牌与本地容器路径前缀，
   同时允许 NLP 笔记中的正常缩写（如 Lv et al.）。所有命中值在返回消息、终端输出与 JSON 回执中均做脱敏（只报文件、行号、规则名与长度，不回显原值）。
3. 白名单精确到具体匹配值（公开 noreply 邮箱、平台路径 /kaggle/working 等），不按整文件豁免；
   共用索引文件与 agents/ 归档照常扫描。
4. 严格 Git 错误检查与暂存区/工作区分离扫描：
   - 任何用于枚举或读取扫描输入的 Git 调用失败（如无效 commit 范围、浅克隆缺 base）均直接报错退出（返回码 1）。
   - 检查未提交改动时，分别枚举并扫描暂存区（git diff --cached 与 index blob :path）和工作区（git diff 与未跟踪文件），防止暂存敏感内容后改回工作区导致漏扫。
   - 覆盖一次 push 范围内的每个 commit 逐个扫描。
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

# 2. 结合私有字段与上下文判定的等级缩写 / 关卡字段（防 NLP 正常缩写误报）
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
    """检查一组改动路径（含新增、修改、删除及重命名源/目标）是否跨越多个非共用工作线。"""
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


def format_redacted_finding(label: str, lineno: int, kind: str, matched_val: str) -> str:
    """生成脱敏告警字符串：只保留文件、行号、规则名称和长度，绝不回显命中的原始凭据或隐私值。"""
    return f"{label}:{lineno} 命中{kind}（已脱敏，长度={len(matched_val)}）"


def scan_text(label: str, text: str) -> list[str]:
    """逐行扫描单个文本内容，返回脱敏后的违规项列表（无整文件豁免）。"""
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for rx, kind in (
            (PRIVATE_ARCHIVE_RE, "私有存档字段"),
            (CONTEXT_LV_RE, "私有等级字段"),
            (CONTEXT_BOSS_RE, "私有Boss字段"),
            (TOKEN_RE, "密钥令牌"),
        ):
            for m in rx.finditer(line):
                findings.append(format_redacted_finding(label, lineno, kind, m.group(0)))

        for m in EMAIL_RE.finditer(line):
            val = m.group(0)
            if val not in ALLOWED_EMAILS:
                findings.append(format_redacted_finding(label, lineno, "非白名单邮箱", val))

        for m in LOCAL_PATH_RE.finditer(line):
            val = m.group(0)
            if not is_allowed_path(val):
                findings.append(format_redacted_finding(label, lineno, "本地或容器路径", val))
    return findings


def git_run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def git_checked(repo: Path, git_errors: list[str], context: str, *args: str) -> subprocess.CompletedProcess[str] | None:
    """执行 Git 命令并严格检查返回码；失败时记入 git_errors。"""
    r = git_run(repo, *args)
    if r.returncode != 0:
        detail = (r.stderr.strip() or r.stdout.strip() or f"exit code {r.returncode}").splitlines()[0]
        git_errors.append(f"{context}失败 (git {' '.join(args)}): {detail}")
        return None
    return r


def parse_name_status(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    """解析 git diff / diff-tree 的 --name-status 输出。

    返回 (ownership_paths, content_paths)：
    - 删除（D）：路径计入 ownership_paths，不计入 content_paths；
    - 重命名（R*）或复制（C*）：源路径和目标路径均计入 ownership_paths，目标路径计入 content_paths；
    - 其他状态（A/M/T 等）：路径同时计入 ownership_paths 和 content_paths。
    """
    ownership_paths: list[str] = []
    content_paths: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0].upper()
        if status.startswith(("R", "C")) and len(parts) >= 3:
            old_p, new_p = parts[1].strip(), parts[2].strip()
            ownership_paths.extend([old_p, new_p])
            content_paths.append(new_p)
        elif status.startswith("D") and len(parts) >= 2:
            old_p = parts[1].strip()
            ownership_paths.append(old_p)
        elif len(parts) >= 2:
            p = parts[1].strip()
            ownership_paths.append(p)
            content_paths.append(p)
    return sorted(set(ownership_paths)), sorted(set(content_paths))


def scan_commit(
    repo: Path, rev: str, git_errors: list[str]
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """扫描单个 commit 的路径归属（含删除与重命名两端）及现存改动文件的完整内容。"""
    r = git_checked(
        repo,
        git_errors,
        f"枚举 commit {rev[:8]} 改动",
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-M",
        "--root",
        rev,
    )
    if r is None:
        return [], [], {}

    ownership_paths, content_paths = parse_name_status(r.stdout.splitlines())
    ownership_issues, by_line = check_ownership(ownership_paths)
    ownership_issues = [f"[{rev[:8]}] {msg}" for msg in ownership_issues]

    privacy_findings: list[str] = []
    for rel_path in content_paths:
        blob = git_checked(repo, git_errors, f"读取 commit {rev[:8]} 文件 {rel_path}", "show", f"{rev}:{rel_path}")
        if blob is not None and "\x00" not in blob.stdout:
            privacy_findings.extend(scan_text(f"{rev[:8]}:{rel_path}", blob.stdout))
    return ownership_issues, privacy_findings, by_line


def scan_uncommitted(
    repo: Path, git_errors: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """分别枚举并扫描暂存区（index）与工作区（worktree），返回 (ownership_issues, privacy_findings, scanned_labels)。"""
    ownership_issues: list[str] = []
    privacy_findings: list[str] = []
    scanned_labels: list[str] = []

    # 1. 暂存区（index vs HEAD）
    staged_res = git_checked(repo, git_errors, "枚举暂存区改动", "diff", "--cached", "--name-status", "-M")
    staged_own: list[str] = []
    staged_content: list[str] = []
    if staged_res is not None:
        staged_own, staged_content = parse_name_status(staged_res.stdout.splitlines())
        own_idx, _ = check_ownership(staged_own)
        ownership_issues.extend(f"[index] {m}" for m in own_idx)
        for rel in staged_content:
            label = f"index:{rel}"
            scanned_labels.append(label)
            blob = git_checked(repo, git_errors, f"读取暂存区 blob {rel}", "show", f":{rel}")
            if blob is not None and "\x00" not in blob.stdout:
                privacy_findings.extend(scan_text(label, blob.stdout))

    # 2. 工作区（worktree vs index）+ 未跟踪文件
    wt_res = git_checked(repo, git_errors, "枚举工作区改动", "diff", "--name-status", "-M")
    untracked_res = git_checked(
        repo, git_errors, "枚举未跟踪文件", "ls-files", "--others", "--exclude-standard"
    )
    wt_own: list[str] = []
    wt_content: list[str] = []
    if wt_res is not None:
        wt_own, wt_content = parse_name_status(wt_res.stdout.splitlines())
    if untracked_res is not None:
        untracked = [l.strip() for l in untracked_res.stdout.splitlines() if l.strip()]
        wt_own = sorted(set(wt_own) | set(untracked))
        wt_content = sorted(set(wt_content) | set(untracked))

    own_wt, _ = check_ownership(wt_own)
    ownership_issues.extend(f"[worktree] {m}" for m in own_wt)

    # 若暂存区与工作区各自单线但合起来跨线，也记录一条组合告警
    if not own_idx_or_wt(ownership_issues) and (staged_own and wt_own):
        own_combined, _ = check_ownership(staged_own + wt_own)
        ownership_issues.extend(f"[uncommitted] {m}" for m in own_combined)

    for rel in wt_content:
        fp = repo / rel
        if fp.is_file():
            label = f"worktree:{rel}"
            scanned_labels.append(label)
            raw = fp.read_bytes()
            if b"\x00" not in raw:
                privacy_findings.extend(scan_text(label, raw.decode("utf-8", errors="replace")))

    return ownership_issues, privacy_findings, scanned_labels


def own_idx_or_wt(issues: list[str]) -> bool:
    return len(issues) > 0


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
    ap.add_argument("--include-worktree", action="store_true", help="同时检查暂存区（index）与工作区（worktree）未提交改动")
    ap.add_argument(
        "--ownership-mode",
        choices=("warn", "error"),
        default="warn",
        help="路径归属跨线时的处理方式（默认 warn 试跑，验收或严格模式用 error）",
    )
    ap.add_argument("--json-receipt", help="输出 JSON 验收回执的路径")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    git_errors: list[str] = []
    ownership_warnings: list[str] = []
    ownership_errors: list[str] = []
    privacy_errors: list[str] = []
    scanned_commits: list[str] = []
    scanned_files: list[str] = []

    if args.files is not None:
        if not args.files:
            git_errors.append("--files 参数不能为空列表")
        rel_list: list[str] = []
        for f in args.files:
            fp = Path(f)
            if not fp.is_absolute():
                fp = repo / fp
            fp = fp.resolve()
            if not fp.is_file():
                git_errors.append(f"指定扫描的文件不存在：{f}")
                continue
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
        chk = git_checked(repo, git_errors, "验证 Git 仓库", "rev-parse", "--git-dir")
        if chk is not None:
            explicit_range = args.commit_range is not None
            rng = args.commit_range if explicit_range else default_commit_range(repo)
            if rng:
                revs_proc = git_checked(repo, git_errors, f"解析 commit 范围 {rng}", "rev-list", "--reverse", rng)
                if revs_proc is not None:
                    revs = [l.strip() for l in revs_proc.stdout.splitlines() if l.strip()]
                    for rev in revs:
                        scanned_commits.append(rev)
                        own_issues, priv_issues, _ = scan_commit(repo, rev, git_errors)
                        if args.ownership_mode == "error":
                            ownership_errors.extend(own_issues)
                        else:
                            ownership_warnings.extend(own_issues)
                        privacy_errors.extend(priv_issues)

            # 仅当显式指定 --include-worktree，或未显式指定 --commit-range 且无待扫 commit 时，扫描暂存区与工作区
            if args.include_worktree or (not explicit_range and not scanned_commits and not git_errors):
                own_issues, priv_issues, labels = scan_uncommitted(repo, git_errors)
                scanned_files.extend(labels)
                if args.ownership_mode == "error":
                    ownership_errors.extend(own_issues)
                else:
                    ownership_warnings.extend(own_issues)
                privacy_errors.extend(priv_issues)

    total_err = len(git_errors) + len(ownership_errors) + len(privacy_errors)
    receipt = {
        "script": "agents/check_repo_hygiene.py",
        "script_sha256": script_sha256(),
        "ownership_mode": args.ownership_mode,
        "scanned_commits": scanned_commits,
        "scanned_files": scanned_files,
        "git_errors": git_errors,
        "ownership_warnings": ownership_warnings,
        "ownership_errors": ownership_errors,
        "privacy_errors": privacy_errors,
        "passed": total_err == 0,
    }
    if args.json_receipt:
        out_path = Path(args.json_receipt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for w in ownership_warnings:
        print(f"提醒：{w}")
    for e in git_errors:
        print(f"错误（Git 调用）：{e}")
    for e in ownership_errors:
        print(f"错误（路径归属）：{e}")
    for e in privacy_errors:
        print(f"错误（隐私/路径）：{e}")

    print(
        f"检查完成：扫描 {len(scanned_commits)} 个 commit、{len(scanned_files)} 个文件；"
        f"{total_err} 个错误，{len(ownership_warnings)} 个提醒。"
    )
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
