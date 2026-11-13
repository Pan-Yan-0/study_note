#!/usr/bin/env python3
"""仓库路径归属、任务范围限定、隐私/本机路径扫描与 Drive 版本绑定回执检查脚本。

满足核心验收要求：
1. 路径归属与任务级范围检查：
   - 单次提交只改同一工作线目录；共用索引（根 README.md、ML_note/实验结果汇总.md）默认允许随同更新；
   - ML_note/实验/SemEval_DiCo-NLI/论文研究/ 按独立工作线判定，且不会被父目录前缀 ML_note/实验/SemEval_DiCo-NLI/ 误放行；
   - 支持 --expected-line、--allowed-prefixes 与 --task-manifest，将“具体任务允许改哪里”交给脚本检查，
     拦截“未跨多条线但改动了其他工作线或超出当前任务允许目录”的越界行为；
   - 归属与范围检查覆盖新增、修改、删除（D）以及重命名（R）/复制（C）的源与目标路径。
2. 上下文感知的隐私与本机路径扫描：
   - 拦截私有学习字段、个人邮箱、密钥令牌与本地容器路径前缀，同时允许 NLP 笔记中的正常缩写（如 Lv et al.）；
   - 所有命中值在返回消息、终端输出、JSON 回执与 Markdown 回执中均做脱敏（只报文件、行号、规则名与长度，不回显原值）。
3. 白名单精确到具体匹配值（公开 noreply 邮箱、平台路径 /kaggle/working 等），不按整文件豁免；
   共用索引文件与 agents/ 归档照常扫描。
4. 严格 Git 错误检查与暂存区/工作区分离扫描：
   - 任何用于枚举或读取扫描输入的 Git 调用失败（如无效 commit 范围、浅克隆缺 base）均直接报错退出（返回码 1）；
   - 检查未提交改动时，分别枚举并扫描暂存区（git diff --cached 与 index blob :path）和工作区（git diff 与未跟踪文件）；
   - 覆盖一次 push 范围内的每个 commit 逐个扫描。
5. Drive 主工作区交接元数据与版本绑定回执：
   - 支持解析并校验 6 项交接必要信息（任务与当前负责人、允许修改范围、来源版本、可见范围、当前状态、正式结果链接）；
   - 支持 --require-public-ready 拦截仍处于“工作区私有”状态的草稿进入公开仓库；
   - 回执绑定文件 SHA-256、Drive fileId 与 Git commit；支持 --verify-receipt 校验文件是否在检查后被再次修改（若修改则旧回执自动失效）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

SHARED_INDEX_FILES = {"README.md", "ML_note/实验结果汇总.md"}
VALID_LINES = ("semeval", "semeval_papers", "learning", "infra", "shared_index", "other")
SEMEVAL_PAPERS_PREFIX = "ML_note/实验/SemEval_DiCo-NLI/论文研究/"

PRIVATE_VISIBILITY_VALUES = {"工作区私有", "PRIVATE_WORKSPACE"}
PUBLIC_VISIBILITY_VALUES = {"允许公开发布", "PUBLIC_READY"}
VALID_VISIBILITY_VALUES = PRIVATE_VISIBILITY_VALUES | PUBLIC_VISIBILITY_VALUES

VALID_STATUS_VALUES = {
    "待提稿",
    "待审",
    "退回",
    "通过",
    "DRAFT",
    "PENDING",
    "RETURNED",
    "APPROVED",
}

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


def normalize_rel_path(rel_path: str) -> str:
    p = rel_path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def classify_line(rel_path: str) -> str:
    """将仓库相对路径映射为工作线名称。"""
    p = normalize_rel_path(rel_path)
    if p in SHARED_INDEX_FILES:
        return "shared_index"
    if p.startswith(SEMEVAL_PAPERS_PREFIX):
        return "semeval_papers"
    if p.startswith("ML_note/实验/SemEval_DiCo-NLI/"):
        return "semeval"
    if p.startswith(("ML_note/原理笔记/", "ML_note/论文阅读/", "ML_note/实验/ShortGPT_BI删层/", "ML_note/吴恩达机器学习/")):
        return "learning"
    if p in {"AGENTS.md", "CLAUDE.md", ".gitignore"} or p.startswith(("agents/", ".github/")):
        return "infra"
    return "other"


def path_matches_allowed_prefixes(rel_path: str, allowed_prefixes: Iterable[str]) -> bool:
    """判断路径是否落在任务允许修改的路径前缀或精确路径内。

    特别保护：若允许前缀为 ML_note/实验/SemEval_DiCo-NLI/（未显式指向 论文研究/），
    不放行属于独立工作线的 ML_note/实验/SemEval_DiCo-NLI/论文研究/。
    """
    p = normalize_rel_path(rel_path)
    for raw_prefix in allowed_prefixes:
        pref = normalize_rel_path(raw_prefix).strip()
        if not pref:
            continue
        # 防止父目录前缀穿透到独立的 论文研究/ 子线
        if p.startswith(SEMEVAL_PAPERS_PREFIX) and not pref.startswith(SEMEVAL_PAPERS_PREFIX.rstrip("/")):
            continue
        if p == pref:
            return True
        if pref.endswith("/") and p.startswith(pref):
            return True
        if not pref.endswith("/") and p.startswith(pref + "/"):
            return True
    return False


def check_ownership(
    paths: Iterable[str],
    expected_line: str | None = None,
    allowed_prefixes: Iterable[str] | None = None,
    allow_shared_index: bool = True,
    task_label: str | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """检查一组改动路径（含新增、修改、删除及重命名源/目标）是否跨越工作线或越出任务允许范围。"""
    by_line: dict[str, list[str]] = {}
    norm_paths = [normalize_rel_path(p) for p in sorted(set(paths)) if p and normalize_rel_path(p)]
    for p in norm_paths:
        line = classify_line(p)
        by_line.setdefault(line, []).append(p)

    non_shared = {k: v for k, v in by_line.items() if k != "shared_index"}
    issues: list[str] = []
    if len(non_shared) > 1:
        summary = ", ".join(f"{k}({len(v)}个文件: {v[0]}等)" for k, v in sorted(non_shared.items()))
        issues.append(f"单次提交跨越多个工作线：{summary}")

    prefix_list = [normalize_rel_path(x) for x in (allowed_prefixes or []) if normalize_rel_path(x)]
    ctx = f"任务[{task_label}]" if task_label else "当前任务"

    if expected_line:
        for p in norm_paths:
            line = classify_line(p)
            if allow_shared_index and line == "shared_index":
                continue
            if line != expected_line:
                issues.append(
                    f"越界修改（工作线不符）：路径 {p} 属于工作线 {line}，不在{ctx}限定的工作线 {expected_line} 内"
                )

    if prefix_list:
        allowed_desc = ", ".join(prefix_list)
        for p in norm_paths:
            if allow_shared_index and p in SHARED_INDEX_FILES:
                continue
            if not path_matches_allowed_prefixes(p, prefix_list):
                issues.append(
                    f"越界修改（超出任务范围）：路径 {p} 不在{ctx}允许修改的范围内（允许范围：{allowed_desc}）"
                )

    return issues, by_line


def _split_prefix_values(raw: str) -> list[str]:
    """从 Markdown 单元格或字符串中提取允许修改的路径列表。"""
    code_spans = re.findall(r"`([^`]+)`", raw)
    if code_spans:
        return [normalize_rel_path(c.strip()) for c in code_spans if c.strip()]
    parts = re.split(r"[,，;；、]+", raw)
    return [normalize_rel_path(p.strip()) for p in parts if p.strip()]


def parse_task_manifest_text(text: str) -> dict[str, Any]:
    """解析 JSON 或 Markdown 顶部交接元数据表，统一返回结构化字段。"""
    stripped = text.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        return {
            "task_id": str(data.get("task_id", "")).strip(),
            "owner": str(data.get("owner", "")).strip(),
            "expected_line": str(data.get("expected_line", "")).strip() or None,
            "allowed_prefixes": [normalize_rel_path(x) for x in data.get("allowed_prefixes", []) if str(x).strip()],
            "allow_shared_index": bool(data.get("allow_shared_index", True)),
            "source_version": str(data.get("source_version", "")).strip(),
            "visibility": str(data.get("visibility", "")).strip(),
            "status": str(data.get("status", "")).strip(),
            "canonical_link": str(data.get("canonical_link", "")).strip(),
            "target_repo_path": normalize_rel_path(str(data.get("target_repo_path", "")).strip())
            if data.get("target_repo_path")
            else "",
        }

    manifest: dict[str, Any] = {
        "task_id": "",
        "owner": "",
        "expected_line": None,
        "allowed_prefixes": [],
        "allow_shared_index": True,
        "source_version": "",
        "visibility": "",
        "status": "",
        "canonical_link": "",
        "target_repo_path": "",
    }

    for line in text.splitlines()[:80]:
        s = line.strip()
        if not s:
            continue
        key, val = "", ""
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|", 1)]
            if len(cells) >= 2 and not set(cells[0].replace("|", "")) <= {"-", ":", " "}:
                key, val = cells[0].strip("* "), cells[1].strip()
        elif ":" in s or "：" in s:
            m = re.match(r"^[-*>]?\s*\*?\*?([^:*：]+)\*?\*?\s*[:：]\s*(.+)$", s)
            if m:
                key, val = m.group(1).strip(), m.group(2).strip()
        if not key or not val:
            continue

        clean_val = val.strip("` ").strip()
        k_lower = key.lower()
        if key in {"任务与当前负责人", "任务与负责人"}:
            if "|" in val or "｜" in val:
                parts = re.split(r"[|｜]", val, maxsplit=1)
                manifest["task_id"] = parts[0].strip("` ")
                manifest["owner"] = parts[1].strip("` ")
            else:
                manifest["task_id"] = clean_val
                manifest["owner"] = clean_val
        elif key in {"任务ID", "任务名", "任务名称"} or k_lower == "task_id":
            manifest["task_id"] = clean_val
        elif key in {"当前负责人", "负责人"} or k_lower == "owner":
            manifest["owner"] = clean_val
        elif key in {"所属工作线", "工作线"} or k_lower == "expected_line":
            manifest["expected_line"] = clean_val
        elif key in {"允许修改范围", "允许修改路径"} or k_lower == "allowed_prefixes":
            manifest["allowed_prefixes"] = _split_prefix_values(val)
        elif key in {"来源版本", "输入版本"} or k_lower == "source_version":
            manifest["source_version"] = clean_val
        elif key in {"可见范围", "公开状态"} or k_lower == "visibility":
            manifest["visibility"] = clean_val
        elif key in {"当前状态", "审阅状态", "状态"} or k_lower == "status":
            manifest["status"] = clean_val
        elif key in {"正式结果链接", "正式链接"} or k_lower == "canonical_link":
            manifest["canonical_link"] = clean_val
        elif key in {"目标仓库路径", "仓库目标路径"} or k_lower == "target_repo_path":
            manifest["target_repo_path"] = normalize_rel_path(clean_val)

    return manifest


def parse_task_manifest(path: Path) -> dict[str, Any]:
    return parse_task_manifest_text(path.read_text(encoding="utf-8", errors="replace"))


def validate_handoff_manifest(
    manifest: dict[str, Any], require_public_ready: bool = False
) -> list[str]:
    """校验交接文件的 6 项必要元数据是否齐全合规。"""
    errors: list[str] = []
    if not (manifest.get("task_id") or manifest.get("owner")):
        errors.append("交接元数据缺失：缺少「任务与当前负责人」")
    if not manifest.get("allowed_prefixes"):
        errors.append("交接元数据缺失：缺少「允许修改范围」")
    if not manifest.get("source_version"):
        errors.append("交接元数据缺失：缺少「来源版本」")
    vis = manifest.get("visibility", "")
    if not vis:
        errors.append("交接元数据缺失：缺少「可见范围」（应为 工作区私有 或 允许公开发布）")
    elif vis not in VALID_VISIBILITY_VALUES:
        errors.append(f"交接元数据无效：「可见范围」值 {vis!r} 不合法（仅允许 工作区私有 或 允许公开发布）")
    elif require_public_ready and vis not in PUBLIC_VISIBILITY_VALUES:
        errors.append(f"公开发布拦截：当前文件可见范围为「{vis}」，未标记为「允许公开发布」，禁止合入公开仓库")

    st = manifest.get("status", "")
    if not st:
        errors.append("交接元数据缺失：缺少「当前状态」（待提稿 / 待审 / 退回 / 通过）")
    elif st not in VALID_STATUS_VALUES:
        errors.append(f"交接元数据无效：「当前状态」值 {st!r} 不合法（允许：待提稿 / 待审 / 退回 / 通过）")

    if not (manifest.get("canonical_link") or manifest.get("target_repo_path")):
        errors.append("交接元数据缺失：缺少「正式结果链接」（或「目标仓库路径」）")
    return errors


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
    repo: Path,
    rev: str,
    git_errors: list[str],
    expected_line: str | None = None,
    allowed_prefixes: Iterable[str] | None = None,
    allow_shared_index: bool = True,
    task_label: str | None = None,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """扫描单个 commit 的路径归属/任务范围（含删除与重命名两端）及现存改动文件的完整内容。"""
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
    ownership_issues, by_line = check_ownership(
        ownership_paths,
        expected_line=expected_line,
        allowed_prefixes=allowed_prefixes,
        allow_shared_index=allow_shared_index,
        task_label=task_label,
    )
    ownership_issues = [f"[{rev[:8]}] {msg}" for msg in ownership_issues]

    privacy_findings: list[str] = []
    for rel_path in content_paths:
        blob = git_checked(repo, git_errors, f"读取 commit {rev[:8]} 文件 {rel_path}", "show", f"{rev}:{rel_path}")
        if blob is not None and "\x00" not in blob.stdout:
            privacy_findings.extend(scan_text(f"{rev[:8]}:{rel_path}", blob.stdout))
    return ownership_issues, privacy_findings, by_line


def scan_uncommitted(
    repo: Path,
    git_errors: list[str],
    expected_line: str | None = None,
    allowed_prefixes: Iterable[str] | None = None,
    allow_shared_index: bool = True,
    task_label: str | None = None,
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
        own_idx, _ = check_ownership(
            staged_own,
            expected_line=expected_line,
            allowed_prefixes=allowed_prefixes,
            allow_shared_index=allow_shared_index,
            task_label=task_label,
        )
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

    own_wt, _ = check_ownership(
        wt_own,
        expected_line=expected_line,
        allowed_prefixes=allowed_prefixes,
        allow_shared_index=allow_shared_index,
        task_label=task_label,
    )
    ownership_issues.extend(f"[worktree] {m}" for m in own_wt)

    # 若暂存区与工作区各自单线但合起来跨线，也记录一条组合告警
    if not own_idx_or_wt(ownership_issues) and (staged_own and wt_own):
        own_combined, _ = check_ownership(
            staged_own + wt_own,
            expected_line=expected_line,
            allowed_prefixes=allowed_prefixes,
            allow_shared_index=allow_shared_index,
            task_label=task_label,
        )
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


def current_head_commit(repo: Path) -> str:
    r = git_run(repo, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else ""


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verify_bound_receipt(
    receipt_path: Path, current_file_hashes: dict[str, str]
) -> list[str]:
    """核对当前待审文件的 SHA-256 是否与已签发回执中绑定的版本完全一致。"""
    if not receipt_path.is_file():
        return [f"回执校验失败：找不到回执文件 {receipt_path}"]
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"回执校验失败：回执 JSON 解析错误 ({exc})"]

    errors: list[str] = []
    if not data.get("passed", False):
        errors.append("回执校验失败：原回执记录的检查结果为未通过（passed=false），不能作为审阅或发布依据")

    bound_hashes: dict[str, str] = data.get("file_hashes", {})
    if not bound_hashes:
        errors.append("回执校验失败：原回执未绑定任何文件 SHA-256（file_hashes 为空）")
        return errors

    for rel, cur_sha in current_file_hashes.items():
        if rel not in bound_hashes:
            errors.append(f"版本失效：文件 {rel} 不在原回执绑定清单中，新稿须重新送检")
        elif bound_hashes[rel] != cur_sha:
            errors.append(
                f"版本失效：文件 {rel} 当前 SHA-256 ({cur_sha[:12]}) 与回执绑定版本 ({bound_hashes[rel][:12]}) 不一致；"
                "旧版本的“通过”只适用于旧版本，新稿必须重新送检"
            )
    return errors


def render_markdown_receipt(receipt: dict[str, Any]) -> str:
    """生成写回 Google Drive 任务目录的 Markdown 版本绑定检查回执。"""
    status_str = "通过（PASSED）" if receipt["passed"] else "未通过（FAILED）"
    lines = [
        "# 自动检查回执（绑定确定版本快照）",
        "",
        "> **版本绑定规则**：本回执仅对下方表内列出的文件 `SHA-256` 快照生效。若写作者（如 Gemini 网页端）在回执生成后再次修改了 Drive 文件，文件内容哈希改变，本回执自动失效，新稿须重新送检后再由审阅端（如 GPT 聊天端）审核。",
        "",
        "| 检查字段 | 记录值 |",
        "|---|---|",
        f"| 检查结论 | **{status_str}** |",
        f"| 检查脚本 | `{receipt['script']}` (`{receipt['script_sha256'][:12]}`) |",
        f"| 基线 Git Commit | `{receipt.get('git_head_commit') or 'N/A'}` |",
        f"| 绑定 Drive File ID | `{receipt.get('drive_file_id') or 'N/A'}` |",
        f"| 任务与负责人 | `{receipt.get('task_label') or 'N/A'}` |",
        f"| 允许修改范围 | `{', '.join(receipt.get('allowed_prefixes') or []) or '按工作线默认规则'}` |",
        f"| 可见范围 | `{receipt.get('visibility') or 'N/A'}` |",
        "",
        "## 绑定文件快照（File SHA-256）",
        "",
    ]
    file_hashes: dict[str, str] = receipt.get("file_hashes", {})
    if file_hashes:
        lines.append("| 仓库目标路径 / 扫描标签 | 内容 SHA-256 |")
        lines.append("|---|---|")
        for rel, sha in sorted(file_hashes.items()):
            lines.append(f"| `{rel}` | `{sha}` |")
    else:
        lines.append("- 本次按 Git commit 范围扫描，未单列外部文件快照。")

    lines.extend([
        "",
        "## 检查明细",
        "",
        f"- 扫描 commit 数：`{len(receipt.get('scanned_commits', []))}`",
        f"- 扫描文件数：`{len(receipt.get('scanned_files', []))}`",
        f"- Git/元数据错误：`{len(receipt.get('git_errors', []))}`",
        f"- 路径归属与任务越界错误：`{len(receipt.get('ownership_errors', []))}`（提醒：`{len(receipt.get('ownership_warnings', []))}`）",
        f"- 隐私与本机路径错误：`{len(receipt.get('privacy_errors', []))}`",
    ])
    all_errs = (
        receipt.get("git_errors", [])
        + receipt.get("ownership_errors", [])
        + receipt.get("privacy_errors", [])
    )
    if all_errs:
        lines.append("")
        lines.append("### 错误列表（已脱敏）")
        for e in all_errs:
            lines.append(f"- {e}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="Git 仓库根目录")
    ap.add_argument("--commit-range", help="要逐个扫描的 commit 范围，如 origin/note..HEAD")
    ap.add_argument("--files", nargs="*", help="直接扫描指定文件（相对仓库根目录或绝对路径）")
    ap.add_argument("--target-rel-path", help="当扫描仓库外部的交接草稿时，指定其拟合入仓库的目标相对路径")
    ap.add_argument("--include-worktree", action="store_true", help="同时检查暂存区（index）与工作区（worktree）未提交改动")
    ap.add_argument(
        "--ownership-mode",
        choices=("warn", "error"),
        default="warn",
        help="路径归属与任务范围越界时的处理方式（默认 warn 试跑，验收或严格模式用 error）",
    )
    ap.add_argument(
        "--expected-line",
        choices=("semeval", "semeval_papers", "learning", "infra"),
        help="限定本次改动所属的唯一工作线（防止只改另一条线但未跨多条线的越界）",
    )
    ap.add_argument(
        "--allowed-prefixes",
        nargs="*",
        help="限定当前任务允许修改的路径前缀或精确文件列表",
    )
    ap.add_argument(
        "--task-manifest",
        help="读取任务范围与交接元数据文件（支持 JSON 或含 6 项元数据的 Markdown 交接文件）",
    )
    ap.add_argument(
        "--require-handoff-header",
        action="store_true",
        help="要求被扫描的交接文件顶部必须包含完整的 6 项交接必要元数据",
    )
    ap.add_argument(
        "--require-public-ready",
        action="store_true",
        help="要求交接文件可见范围必须已标记为「允许公开发布」（用于合入公开仓库 study_note 前拦截）",
    )
    ap.add_argument("--drive-file-id", help="记录在回执中的 Google Drive 文件 ID")
    ap.add_argument("--verify-receipt", help="核对当前待审文件的 SHA-256 是否仍与指定 JSON 回执绑定的版本一致")
    ap.add_argument("--json-receipt", help="输出 JSON 验收回执的路径")
    ap.add_argument("--markdown-receipt", help="输出写回 Drive 的 Markdown 验收回执路径")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    git_errors: list[str] = []
    ownership_warnings: list[str] = []
    ownership_errors: list[str] = []
    privacy_errors: list[str] = []
    scanned_commits: list[str] = []
    scanned_files: list[str] = []
    file_hashes: dict[str, str] = {}

    expected_line = args.expected_line
    allowed_prefixes: list[str] = list(args.allowed_prefixes or [])
    allow_shared_index = True
    task_label: str | None = None
    visibility: str = ""
    manifest_target_rel: str = ""

    # 若指定了 --task-manifest，解析并合并任务范围与交接元数据
    if args.task_manifest:
        tm_path = Path(args.task_manifest)
        if not tm_path.is_file():
            git_errors.append(f"指定的任务清单文件不存在：{args.task_manifest}")
        else:
            try:
                manifest = parse_task_manifest(tm_path)
                if args.require_handoff_header or args.require_public_ready:
                    git_errors.extend(validate_handoff_manifest(manifest, require_public_ready=args.require_public_ready))
                elif manifest.get("visibility") or manifest.get("status"):
                    # 若文件声明了交接字段，校验其合法性
                    git_errors.extend(validate_handoff_manifest(manifest, require_public_ready=args.require_public_ready))
                if not expected_line and manifest.get("expected_line"):
                    expected_line = manifest["expected_line"]
                if not allowed_prefixes and manifest.get("allowed_prefixes"):
                    allowed_prefixes = list(manifest["allowed_prefixes"])
                allow_shared_index = bool(manifest.get("allow_shared_index", True))
                if manifest.get("task_id") or manifest.get("owner"):
                    task_label = " | ".join(x for x in (manifest.get("task_id"), manifest.get("owner")) if x)
                visibility = manifest.get("visibility", "")
                manifest_target_rel = manifest.get("target_repo_path", "")
            except Exception as exc:
                git_errors.append(f"解析任务清单 {args.task_manifest} 失败：{exc}")

    # 当显式限定了任务范围（expected_line / allowed_prefixes / task_manifest）时，越界默认升级为 error
    effective_ownership_mode = (
        "error"
        if (args.ownership_mode == "error" or expected_line or allowed_prefixes or args.task_manifest)
        else "warn"
    )

    if args.files is not None:
        if not args.files:
            git_errors.append("--files 参数不能为空列表")
        rel_list: list[str] = []
        for idx, f in enumerate(args.files):
            fp = Path(f)
            if not fp.is_absolute():
                fp = repo / fp
            fp = fp.resolve()
            if not fp.is_file():
                git_errors.append(f"指定扫描的文件不存在：{f}")
                continue
            raw_bytes = fp.read_bytes()
            sha = hashlib.sha256(raw_bytes).hexdigest()
            text = raw_bytes.decode("utf-8", errors="replace")

            # 若未单独传 --task-manifest，但开启了 --require-handoff-header 或 --require-public-ready，从文件自身解析
            if not args.task_manifest and (args.require_handoff_header or args.require_public_ready):
                inline_manifest = parse_task_manifest_text(text)
                git_errors.extend(
                    validate_handoff_manifest(inline_manifest, require_public_ready=args.require_public_ready)
                )
                if not allowed_prefixes and inline_manifest.get("allowed_prefixes"):
                    allowed_prefixes = list(inline_manifest["allowed_prefixes"])
                if not expected_line and inline_manifest.get("expected_line"):
                    expected_line = inline_manifest["expected_line"]
                if not task_label and (inline_manifest.get("task_id") or inline_manifest.get("owner")):
                    task_label = " | ".join(
                        x for x in (inline_manifest.get("task_id"), inline_manifest.get("owner")) if x
                    )
                if not visibility:
                    visibility = inline_manifest.get("visibility", "")
                if not manifest_target_rel and inline_manifest.get("target_repo_path"):
                    manifest_target_rel = inline_manifest["target_repo_path"]

            if args.target_rel_path and idx == 0:
                rel = normalize_rel_path(args.target_rel_path)
            elif manifest_target_rel and idx == 0:
                rel = normalize_rel_path(manifest_target_rel)
            else:
                try:
                    rel = normalize_rel_path(str(fp.relative_to(repo)))
                except ValueError:
                    rel = fp.name

            rel_list.append(rel)
            scanned_files.append(rel)
            file_hashes[rel] = sha
            privacy_errors.extend(scan_text(rel, text))

        if expected_line or allowed_prefixes or args.require_handoff_header or args.require_public_ready:
            effective_ownership_mode = "error"

        own_issues, _ = check_ownership(
            rel_list,
            expected_line=expected_line,
            allowed_prefixes=allowed_prefixes,
            allow_shared_index=allow_shared_index,
            task_label=task_label,
        )
        if effective_ownership_mode == "error":
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
                        own_issues, priv_issues, _ = scan_commit(
                            repo,
                            rev,
                            git_errors,
                            expected_line=expected_line,
                            allowed_prefixes=allowed_prefixes,
                            allow_shared_index=allow_shared_index,
                            task_label=task_label,
                        )
                        if effective_ownership_mode == "error":
                            ownership_errors.extend(own_issues)
                        else:
                            ownership_warnings.extend(own_issues)
                        privacy_errors.extend(priv_issues)

            # 仅当显式指定 --include-worktree，或未显式指定 --commit-range 且无待扫 commit 时，扫描暂存区与工作区
            if args.include_worktree or (not explicit_range and not scanned_commits and not git_errors):
                own_issues, priv_issues, labels = scan_uncommitted(
                    repo,
                    git_errors,
                    expected_line=expected_line,
                    allowed_prefixes=allowed_prefixes,
                    allow_shared_index=allow_shared_index,
                    task_label=task_label,
                )
                scanned_files.extend(labels)
                if effective_ownership_mode == "error":
                    ownership_errors.extend(own_issues)
                else:
                    ownership_warnings.extend(own_issues)
                privacy_errors.extend(priv_issues)

    if args.verify_receipt:
        git_errors.extend(verify_bound_receipt(Path(args.verify_receipt), file_hashes))

    total_err = len(git_errors) + len(ownership_errors) + len(privacy_errors)
    receipt = {
        "script": "agents/check_repo_hygiene.py",
        "script_sha256": script_sha256(),
        "git_head_commit": current_head_commit(repo),
        "drive_file_id": args.drive_file_id or "",
        "task_label": task_label or "",
        "expected_line": expected_line or "",
        "allowed_prefixes": allowed_prefixes,
        "visibility": visibility,
        "ownership_mode": effective_ownership_mode,
        "scanned_commits": scanned_commits,
        "scanned_files": scanned_files,
        "file_hashes": file_hashes,
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

    if args.markdown_receipt:
        md_path = Path(args.markdown_receipt)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown_receipt(receipt), encoding="utf-8")

    for w in ownership_warnings:
        print(f"提醒：{w}")
    for e in git_errors:
        print(f"错误（Git/元数据/回执）：{e}")
    for e in ownership_errors:
        print(f"错误（路径归属/任务范围）：{e}")
    for e in privacy_errors:
        print(f"错误（隐私/路径）：{e}")

    print(
        f"检查完成：扫描 {len(scanned_commits)} 个 commit、{len(scanned_files)} 个文件；"
        f"{total_err} 个错误，{len(ownership_warnings)} 个提醒。"
    )
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
