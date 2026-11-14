#!/usr/bin/env python3
"""仓库路径归属、任务范围限定、隐私/本机路径扫描与 Drive 版本绑定回执检查脚本。

满足核心验收要求：
1. 路径归属与活跃任务列表范围检查：
   - 单次提交只改同一工作线目录；共用索引（根 README.md、ML_note/实验结果汇总.md）允许随同更新；
   - 任务绑定清单（agents/current_task.json）不作为共用索引，修改时必须单独作为一个 commit 提交（禁止与其他路径混交）；
   - 扫描非绑定 commit C 时，强制从父提交 C^:agents/current_task.json 读取 active_tasks（扫描未提交改动时从 HEAD 读取），
     只要单次提交全部改动落在至少一个活跃任务的 allowed_prefixes 内且未触碰 forbidden_prefixes 即放行；
   - ML_note/实验/SemEval_DiCo-NLI/论文研究/ 按独立工作线判定，且不会被父目录前缀 ML_note/实验/SemEval_DiCo-NLI/ 误放行；
   - 支持 --expected-line、--allowed-prefixes、--forbidden-prefixes、--task-manifest、--require-task-scope 与 --check-task-pool；
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
5. 公开发布门禁、任务级分端信息共享池校验与 Drive 版本绑定回执：
   - --require-public-ready 同时作用于 --files、commit 范围扫描与暂存区/工作区扫描：任何标记为“工作区私有”或状态为“待审/退回/待提稿”的文件均报错拦截；
   - --check-task-pool（并在 --require-task-scope 下自动校验 agents/task_pools/*.json）核验分端信息共享池 JSON 结构及 github_repo=synced 产物的仓库跟踪完整性；
   - verify_bound_receipt 拒绝空输入，强制要求传入待审文件的实际字节哈希并双向核对回执绑定清单；
   - 支持联动运行 check_archive.py 并将双检查器身份与结果共同写入回执；回执明确区分“格式与卫生检查通过（内容待审）”与“公开发布门禁通过”。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Iterable

TASK_BINDING_REL_PATH = "agents/current_task.json"
TASK_POOLS_REL_PREFIX = "agents/task_pools/"
SHARED_INDEX_FILES = {"README.md", "ML_note/实验结果汇总.md"}
VALID_LINES = ("semeval", "semeval_papers", "learning", "infra", "shared_index", "other")
SEMEVAL_PAPERS_PREFIX = "ML_note/实验/SemEval_DiCo-NLI/论文研究/"

PRIVATE_VISIBILITY_VALUES = {"工作区私有", "PRIVATE_WORKSPACE"}
PUBLIC_VISIBILITY_VALUES = {"允许公开发布", "PUBLIC_READY"}
VALID_VISIBILITY_VALUES = PRIVATE_VISIBILITY_VALUES | PUBLIC_VISIBILITY_VALUES

UNAPPROVED_STATUS_VALUES = {"待提稿", "待审", "退回", "DRAFT", "PENDING", "RETURNED"}
APPROVED_STATUS_VALUES = {"通过", "APPROVED"}
VALID_STATUS_VALUES = UNAPPROVED_STATUS_VALUES | APPROVED_STATUS_VALUES

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


def _matches_dir_prefix(rel_path: str, dir_prefix: str) -> bool:
    base = dir_prefix.rstrip("/")
    return rel_path == base or rel_path.startswith(base + "/")


def classify_line(rel_path: str) -> str:
    """将仓库相对路径映射为工作线名称。"""
    p = normalize_rel_path(rel_path)
    if p in {"README.md", "ML_note/实验结果汇总.md"}:
        return "shared_index"
    if _matches_dir_prefix(p, SEMEVAL_PAPERS_PREFIX):
        return "semeval_papers"
    if _matches_dir_prefix(p, "ML_note/实验/SemEval_DiCo-NLI/"):
        return "semeval"
    if any(
        _matches_dir_prefix(p, pref)
        for pref in ("ML_note/原理笔记/", "ML_note/论文阅读/", "ML_note/实验/ShortGPT_BI删层/", "ML_note/吴恩达机器学习/")
    ):
        return "learning"
    if p in {"AGENTS.md", "CLAUDE.md", ".gitignore"} or any(_matches_dir_prefix(p, pref) for pref in ("agents/", ".github/")):
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
        if _matches_dir_prefix(p, SEMEVAL_PAPERS_PREFIX) and not _matches_dir_prefix(pref, SEMEVAL_PAPERS_PREFIX):
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
    forbidden_prefixes: Iterable[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """检查一组改动路径（含新增、修改、删除及重命名源/目标）是否跨越工作线、违规混交任务绑定文件或越出任务允许范围。"""
    by_line: dict[str, list[str]] = {}
    norm_paths = [normalize_rel_path(p) for p in sorted(set(paths)) if p and normalize_rel_path(p)]
    for p in norm_paths:
        line = classify_line(p)
        by_line.setdefault(line, []).append(p)

    issues: list[str] = []
    if TASK_BINDING_REL_PATH in norm_paths and len(norm_paths) > 1:
        others = ", ".join(p for p in norm_paths if p != TASK_BINDING_REL_PATH)
        issues.append(
            f"越界修改（任务绑定文件违规混交）：{TASK_BINDING_REL_PATH} 必须单独提交，禁止与其他路径（{others}）混在同一个 commit 中修改"
        )

    non_shared = {k: v for k, v in by_line.items() if k != "shared_index"}
    if len(non_shared) > 1:
        summary = ", ".join(f"{k}({len(v)}个文件: {v[0]}等)" for k, v in sorted(non_shared.items()))
        issues.append(f"单次提交跨越多个工作线：{summary}")

    prefix_list = [normalize_rel_path(x) for x in (allowed_prefixes or []) if normalize_rel_path(x)]
    forbidden_list = [normalize_rel_path(x) for x in (forbidden_prefixes or []) if normalize_rel_path(x)]
    ctx = f"任务[{task_label}]" if task_label else "当前任务"

    if expected_line:
        for p in norm_paths:
            if allow_shared_index and p in SHARED_INDEX_FILES:
                continue
            line = classify_line(p)
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

    if forbidden_list:
        forb_desc = ", ".join(forbidden_list)
        for p in norm_paths:
            if path_matches_allowed_prefixes(p, forbidden_list):
                issues.append(
                    f"越界修改（命中禁止修改路径）：路径 {p} 落在{ctx}禁止修改范围（{forb_desc}）内"
                )

    return issues, by_line


def _task_public_ready_error(task: dict[str, Any]) -> str | None:
    """在公开发布门禁下校验匹配任务自身的可见范围与状态，防删去文件头后绕过拦截。"""
    tid = task.get("task_id") or task.get("owner") or "当前任务"
    vis = str(task.get("visibility", "")).strip()
    st = str(task.get("status", "")).strip()
    exp_line = str(task.get("expected_line", "")).strip()
    if vis and vis in PRIVATE_VISIBILITY_VALUES:
        return f"公开发布拦截：匹配任务[{tid}]当前可见范围为「{vis}」，未标记为「允许公开发布」，禁止合入公开仓库"
    if exp_line != "infra" and st and st in UNAPPROVED_STATUS_VALUES:
        return f"公开发布拦截：匹配任务[{tid}]当前状态为「{st}」，尚未审阅通过，禁止合入公开仓库"
    return None


def _check_matched_tasks_public_ready(
    matches: list[dict[str, Any]],
    norm_paths: list[str],
    require_public_ready: bool,
) -> list[str]:
    if not require_public_ready or not matches or not norm_paths or norm_paths == [TASK_BINDING_REL_PATH]:
        return []
    errs: list[str] = []
    for task in matches:
        err = _task_public_ready_error(task)
        if err is None:
            return []
        errs.append(err)
    return [errs[0]] if errs else []


def find_matching_tasks(
    paths: Iterable[str],
    active_tasks: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]], list[dict[str, Any]]]:
    """核对改动路径是否完整落在 active_tasks 中至少一个活跃任务的允许范围内，并返回全部匹配任务。"""
    base_issues, by_line = check_ownership(paths)
    norm_paths = [normalize_rel_path(p) for p in sorted(set(paths)) if p and normalize_rel_path(p)]
    if base_issues or not norm_paths or not active_tasks:
        return base_issues, by_line, []

    matches: list[dict[str, Any]] = []
    for task in active_tasks:
        t_label = " | ".join(x for x in (task.get("task_id"), task.get("owner")) if x) or None
        t_issues, _ = check_ownership(
            norm_paths,
            expected_line=task.get("expected_line"),
            allowed_prefixes=task.get("allowed_prefixes"),
            allow_shared_index=bool(task.get("allow_shared_index", True)),
            task_label=t_label,
            forbidden_prefixes=task.get("forbidden_prefixes"),
        )
        if not t_issues:
            matches.append(task)
    if matches:
        return [], by_line, matches

    non_shared_lines = [k for k in by_line if k != "shared_index"]
    candidate_tasks = [t for t in active_tasks if t.get("expected_line") in non_shared_lines] or active_tasks
    collected: list[str] = []
    for task in candidate_tasks:
        t_label = " | ".join(x for x in (task.get("task_id"), task.get("owner")) if x) or None
        t_issues, _ = check_ownership(
            norm_paths,
            expected_line=task.get("expected_line"),
            allowed_prefixes=task.get("allowed_prefixes"),
            allow_shared_index=bool(task.get("allow_shared_index", True)),
            task_label=t_label,
            forbidden_prefixes=task.get("forbidden_prefixes"),
        )
        for iss in t_issues:
            if iss not in collected:
                collected.append(iss)
    return collected, by_line, []


def check_ownership_against_tasks(
    paths: Iterable[str],
    active_tasks: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]], str | None]:
    """核对一组改动路径是否完整落在 active_tasks 中至少一个活跃任务的允许范围内（且未触碰该任务的 forbidden_prefixes）。"""
    issues, by_line, matches = find_matching_tasks(paths, active_tasks)
    if not matches:
        return issues, by_line, None
    preferred = next((t for t in matches if _task_public_ready_error(t) is None), matches[0])
    t_label = " | ".join(x for x in (preferred.get("task_id"), preferred.get("owner")) if x) or None
    return [], by_line, t_label


def _split_prefix_values(raw: str) -> list[str]:
    """从 Markdown 单元格或字符串中提取允许修改的路径列表。"""
    code_spans = re.findall(r"`([^`]+)`", raw)
    if code_spans:
        return [normalize_rel_path(c.strip()) for c in code_spans if c.strip()]
    parts = re.split(r"[,，;；、]+", raw)
    return [normalize_rel_path(p.strip()) for p in parts if p.strip()]


def normalize_task_entry(raw: dict[str, Any], strict_schema: bool = False) -> dict[str, Any]:
    """规范化并校验单条任务绑定记录（支持 expected_line/workline、allowed_prefixes/allowed_paths、forbidden_prefixes）。"""
    if not isinstance(raw, dict):
        raise ValueError("任务绑定条目必须是 JSON 对象")
    task_id = str(raw.get("task_id", "")).strip()
    owner = str(raw.get("owner", "")).strip()
    expected_line = str(raw.get("expected_line") or raw.get("workline") or "").strip() or None

    raw_prefs = raw.get("allowed_prefixes") if "allowed_prefixes" in raw else raw.get("allowed_paths", [])
    if not isinstance(raw_prefs, list):
        raise ValueError(f"任务 {task_id or '?'} 的 allowed_prefixes 必须是列表")
    allowed_prefixes = [normalize_rel_path(str(x)) for x in raw_prefs if normalize_rel_path(str(x))]

    raw_forb = raw.get("forbidden_prefixes") if "forbidden_prefixes" in raw else raw.get("excluded_prefixes", [])
    if not isinstance(raw_forb, list):
        raise ValueError(f"任务 {task_id or '?'} 的 forbidden_prefixes 必须是列表")
    forbidden_prefixes = [normalize_rel_path(str(x)) for x in raw_forb if normalize_rel_path(str(x))]

    valid_worklines = {"semeval", "semeval_papers", "learning", "infra"}
    if expected_line and expected_line not in valid_worklines:
        raise ValueError(f"任务 {task_id or '?'} 的 expected_line {expected_line!r} 不合法")

    if strict_schema:
        if not task_id or not owner:
            raise ValueError("任务绑定条目必须包含非空 task_id 与 owner")
        if not expected_line:
            raise ValueError(f"任务 {task_id} 缺少 expected_line（或 workline）")
        if not allowed_prefixes:
            raise ValueError(f"任务 {task_id} 的 allowed_prefixes 不能为空")
        for pref in allowed_prefixes:
            if pref in {"", "."}:
                raise ValueError(f"任务 {task_id} 包含非法通配根路径前缀 {pref!r}")
            pref_line = classify_line(pref)
            if pref_line != expected_line:
                raise ValueError(
                    f"任务 {task_id}（工作线 {expected_line}）包含跨工作线路径前缀 {pref!r}（归属 {pref_line}）"
                )
    else:
        if not (expected_line or allowed_prefixes):
            raise ValueError(f"任务 {task_id or '?'} 缺少 expected_line 和 allowed_prefixes")

    return {
        "task_id": task_id,
        "owner": owner,
        "expected_line": expected_line,
        "allowed_prefixes": allowed_prefixes,
        "forbidden_prefixes": forbidden_prefixes,
        "allow_shared_index": bool(raw.get("allow_shared_index", True)),
        "pool_path": normalize_rel_path(str(raw.get("pool_path", "")).strip()) if raw.get("pool_path") else "",
        "source_version": str(raw.get("source_version", "")).strip(),
        "visibility": str(raw.get("visibility", "")).strip(),
        "status": str(raw.get("status", "")).strip(),
        "canonical_link": str(raw.get("canonical_link", "")).strip(),
        "target_repo_path": normalize_rel_path(str(raw.get("target_repo_path", "")).strip())
        if raw.get("target_repo_path")
        else "",
    }


def parse_task_binding_text(text: str, strict_schema: bool = False) -> list[dict[str, Any]]:
    """解析 agents/current_task.json 文本，支持 schema v2.0 active_tasks 列表与旧版单任务字典。"""
    stripped = text.strip()
    if not stripped:
        raise ValueError("任务绑定文件内容为空")
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("任务绑定文件顶层必须是 JSON 对象")
    if "active_tasks" in data:
        raw_tasks = data.get("active_tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("active_tasks 必须是非空列表")
        return [normalize_task_entry(item, strict_schema=strict_schema) for item in raw_tasks]
    return [normalize_task_entry(data, strict_schema=strict_schema)]


_HANDOFF_JSON_STRUCTURAL_KEYS = {
    "task_id",
    "owner",
    "allowed_prefixes",
    "allowed_paths",
    "expected_line",
    "workline",
    "visibility",
    "source_version",
    "canonical_link",
    "target_repo_path",
    "pool_path",
}
_HANDOFF_JSON_KEYS = _HANDOFF_JSON_STRUCTURAL_KEYS | {
    "active_tasks",
    "status",
    "forbidden_prefixes",
    "excluded_prefixes",
    "allow_shared_index",
}
_CHINESE_UNAPPROVED_STATUS_VALUES = {"待提稿", "待审", "退回"}


def _empty_manifest() -> dict[str, Any]:
    return {
        "task_id": "",
        "owner": "",
        "expected_line": None,
        "allowed_prefixes": [],
        "forbidden_prefixes": [],
        "allow_shared_index": True,
        "pool_path": "",
        "source_version": "",
        "visibility": "",
        "status": "",
        "canonical_link": "",
        "target_repo_path": "",
    }


def _is_handoff_json(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    keys = set(data.keys())
    if "active_tasks" in keys or "visibility" in keys:
        return True
    return len(keys & _HANDOFF_JSON_KEYS) >= 2 and bool(keys & _HANDOFF_JSON_STRUCTURAL_KEYS)


def _split_md_table_row(line: str) -> list[str]:
    """按表格外（行内反引号之外且非转义）的 '|' 拆分 Markdown 表格行。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells: list[str] = []
    cur: list[str] = []
    in_code = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n and not in_code:
            cur.append(ch)
            cur.append(s[i + 1])
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def parse_task_manifest_text(
    text: str, require_handoff_json_signature: bool = False
) -> dict[str, Any]:
    """解析 JSON 或 Markdown 交接元数据表，统一返回结构化字段。"""
    stripped = text.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        if not isinstance(data, dict):
            if require_handoff_json_signature:
                return _empty_manifest()
            raise ValueError("任务清单 JSON 顶层必须是对象")
        if require_handoff_json_signature and not _is_handoff_json(data):
            out = _empty_manifest()
            raw_st = str(data.get("status", "")).strip()
            if raw_st in _CHINESE_UNAPPROVED_STATUS_VALUES:
                out["status"] = raw_st
            return out
        if "active_tasks" in data:
            tasks = parse_task_binding_text(stripped, strict_schema=False)
            first = tasks[0]
            return {
                **first,
                "active_tasks": tasks,
            }
        return {
            "task_id": str(data.get("task_id", "")).strip(),
            "owner": str(data.get("owner", "")).strip(),
            "expected_line": str(data.get("expected_line") or data.get("workline") or "").strip() or None,
            "allowed_prefixes": [normalize_rel_path(x) for x in data.get("allowed_prefixes", []) if str(x).strip()],
            "forbidden_prefixes": [normalize_rel_path(x) for x in data.get("forbidden_prefixes", []) if str(x).strip()],
            "allow_shared_index": bool(data.get("allow_shared_index", True)),
            "pool_path": normalize_rel_path(str(data.get("pool_path", "")).strip()) if data.get("pool_path") else "",
            "source_version": str(data.get("source_version", "")).strip(),
            "visibility": str(data.get("visibility", "")).strip(),
            "status": str(data.get("status", "")).strip(),
            "canonical_link": str(data.get("canonical_link", "")).strip(),
            "target_repo_path": normalize_rel_path(str(data.get("target_repo_path", "")).strip())
            if data.get("target_repo_path")
            else "",
        }

    manifest = _empty_manifest()

    for line in text.splitlines()[:200]:
        s = line.strip()
        if not s:
            continue
        key, val = "", ""
        if s.startswith("|") and s.endswith("|"):
            cells = _split_md_table_row(s)
            if len(cells) == 2 and not set(cells[0].replace("|", "")) <= {"-", ":", " "}:
                key, val = cells[0].strip("* "), cells[1].strip()
        elif ":" in s or "：" in s:
            m = re.match(r"^[-*>]?\s*\*?\*?([^:*：|]+)\*?\*?\s*[:：]\s*(.+)$", s)
            if m:
                key, val = m.group(1).strip(), m.group(2).strip()
        if not key or not val:
            continue

        clean_val = val.strip("` ").strip()
        is_table_row = s.startswith("|")
        k_lower = key.lower() if is_table_row else ""
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
        elif key in {"禁止修改范围", "禁止修改路径"} or k_lower == "forbidden_prefixes":
            manifest["forbidden_prefixes"] = _split_prefix_values(val)
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
    return parse_task_manifest_text(
        path.read_text(encoding="utf-8", errors="replace"),
        require_handoff_json_signature=False,
    )


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
    elif require_public_ready and st in UNAPPROVED_STATUS_VALUES:
        errors.append(f"公开发布拦截：当前文件状态为「{st}」，尚未审阅通过，禁止合入公开仓库")

    if not (manifest.get("canonical_link") or manifest.get("target_repo_path")):
        errors.append("交接元数据缺失：缺少「正式结果链接」（或「目标仓库路径」）")
    return errors


def check_content_gates(
    label: str,
    text: str,
    require_handoff_header: bool = False,
    require_public_ready: bool = False,
) -> list[str]:
    """对文件内容执行交接元数据校验与公开发布门禁校验（同时适用于 --files、commit 扫描与暂存区/工作区扫描）。"""
    if not (require_handoff_header or require_public_ready):
        return []
    # 任务绑定 JSON 与任务池 JSON 本身用于记录多端状态（由专门的 schema 校验器检查），不按交接文稿头拦截
    norm_lbl = normalize_rel_path(label.split(":", 1)[-1] if ":" in label else label)
    if norm_lbl == TASK_BINDING_REL_PATH or norm_lbl.startswith(TASK_POOLS_REL_PREFIX):
        return []

    try:
        manifest = parse_task_manifest_text(
            text, require_handoff_json_signature=not require_handoff_header
        )
    except Exception:
        manifest = {}

    errors: list[str] = []
    if require_handoff_header:
        for e in validate_handoff_manifest(manifest, require_public_ready=require_public_ready):
            errors.append(f"{label} {e}")
        return errors

    if require_public_ready:
        vis = manifest.get("visibility", "")
        st = manifest.get("status", "")
        handoff_field_count = sum(
            1
            for k in (
                "task_id",
                "owner",
                "allowed_prefixes",
                "source_version",
                "visibility",
                "status",
                "canonical_link",
                "target_repo_path",
            )
            if manifest.get(k)
        )
        has_strong_handoff = bool(
            manifest.get("task_id")
            or manifest.get("allowed_prefixes")
            or manifest.get("target_repo_path")
            or handoff_field_count >= 2
        )
        if vis:
            if vis not in VALID_VISIBILITY_VALUES:
                errors.append(f"{label} 交接元数据无效：「可见范围」值 {vis!r} 不合法")
            elif vis not in PUBLIC_VISIBILITY_VALUES:
                errors.append(f"{label} 公开发布拦截：当前文件可见范围为「{vis}」，未标记为「允许公开发布」，禁止合入公开仓库")
        elif has_strong_handoff:
            errors.append(f"{label} 公开发布拦截：交接文件缺少「可见范围: 允许公开发布」标记，禁止合入公开仓库")

        if st and st in UNAPPROVED_STATUS_VALUES:
            errors.append(f"{label} 公开发布拦截：当前文件状态为「{st}」，尚未审阅通过，禁止合入公开仓库")

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
    """解析 git diff / diff-tree 的 --name-status 输出。"""
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


def load_repo_active_tasks(
    repo: Path,
    rev_spec: str = "HEAD",
    allow_disk_fallback_if_no_head: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """从指定提交（如 C^ 或 HEAD）读取并解析 agents/current_task.json 的活跃任务列表。"""
    r = git_run(repo, "show", f"{rev_spec}:{TASK_BINDING_REL_PATH}")
    raw_text = ""
    if r.returncode == 0 and r.stdout.strip():
        raw_text = r.stdout
    elif allow_disk_fallback_if_no_head:
        has_head = git_run(repo, "rev-parse", "--verify", "-q", "HEAD").returncode == 0
        if not has_head:
            disk_path = repo / TASK_BINDING_REL_PATH
            if disk_path.is_file():
                raw_text = disk_path.read_text(encoding="utf-8", errors="replace")

    if not raw_text:
        return [], None
    try:
        tasks = parse_task_binding_text(raw_text, strict_schema=False)
        return tasks, None
    except Exception as exc:
        return [], str(exc)


def validate_task_pools(
    repo: Path,
    active_tasks: list[dict[str, Any]] | None = None,
    require_non_empty: bool = False,
) -> list[str]:
    """校验仓库 agents/task_pools/*.json 的任务级分端信息共享池结构，并拦截 github_repo=synced 的假同步。"""
    errors: list[str] = []
    pools_dir = repo / "agents" / "task_pools"
    pool_files: list[Path] = sorted(pools_dir.glob("*.json")) if pools_dir.is_dir() else []
    expected_pool_tasks: dict[str, dict[str, Any]] = {}

    for task in active_tasks or []:
        pp_rel = task.get("pool_path", "")
        if pp_rel:
            norm_pp = normalize_rel_path(pp_rel)
            expected_pool_tasks[norm_pp] = task
            pp = repo / norm_pp
            if not pp.is_file():
                errors.append(
                    f"任务信息共享池缺失：任务 {task.get('task_id') or '?'} 声明的 pool_path {pp_rel} 不存在"
                )
            elif pp not in pool_files:
                pool_files.append(pp)

    if not pool_files:
        if require_non_empty:
            errors.append("任务信息共享池校验失败：未找到任何 agents/task_pools/*.json 文件")
        return errors

    valid_lines = {"semeval", "semeval_papers", "learning", "infra"}
    active_by_id: dict[str, dict[str, Any]] = {
        str(t.get("task_id", "")).strip(): t
        for t in (active_tasks or [])
        if str(t.get("task_id", "")).strip()
    }
    seen_task_ids: dict[str, str] = {}

    for pf in pool_files:
        rel_pf = normalize_rel_path(str(pf.relative_to(repo))) if pf.is_relative_to(repo) else pf.name
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"任务信息共享池 JSON 解析失败 ({rel_pf})：{exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"任务信息共享池格式错误 ({rel_pf})：顶层必须是 JSON 对象")
            continue

        task_id = str(data.get("task_id", "")).strip()
        exp_line = str(data.get("expected_line") or data.get("workline") or "").strip()
        owner = str(data.get("owner", "")).strip()
        endpoints = data.get("endpoints")
        artifacts = data.get("artifacts")

        if not task_id or not owner or exp_line not in valid_lines:
            errors.append(
                f"任务信息共享池元数据缺失 ({rel_pf})：必须包含非空 task_id、owner 及合法的 expected_line"
            )

        if task_id:
            if task_id in seen_task_ids:
                errors.append(
                    f"任务信息共享池重复绑定 ({rel_pf})：task_id={task_id!r} 已在 {seen_task_ids[task_id]} 中定义"
                )
            else:
                seen_task_ids[task_id] = rel_pf

        linked_task = expected_pool_tasks.get(rel_pf)
        if active_by_id and task_id:
            if task_id not in active_by_id:
                errors.append(
                    f"任务信息共享池孤立文件 ({rel_pf})：task_id={task_id!r} 未列入当前活跃任务清单 active_tasks"
                )
            elif not linked_task:
                errors.append(
                    f"任务信息共享池绑定不一致 ({rel_pf})：任务清单中的 {task_id} 未通过 pool_path 指向本文件"
                )

        if linked_task:
            if linked_task.get("task_id") and task_id != linked_task["task_id"]:
                errors.append(
                    f"任务信息共享池绑定不一致 ({rel_pf})：pool task_id={task_id!r} 与任务清单 {linked_task['task_id']!r} 不一致"
                )
            if linked_task.get("expected_line") and exp_line != linked_task["expected_line"]:
                errors.append(
                    f"任务信息共享池绑定不一致 ({rel_pf})：pool expected_line={exp_line!r} 与任务清单 {linked_task['expected_line']!r} 不一致"
                )
            if linked_task.get("owner") and owner and owner != linked_task["owner"]:
                errors.append(
                    f"任务信息共享池绑定不一致 ({rel_pf})：pool owner={owner!r} 与任务清单 {linked_task['owner']!r} 不一致"
                )
            if "allowed_prefixes" in data and linked_task.get("allowed_prefixes"):
                pool_allowed = [normalize_rel_path(str(x)) for x in (data.get("allowed_prefixes") or []) if normalize_rel_path(str(x))]
                if set(pool_allowed) != set(linked_task["allowed_prefixes"]):
                    errors.append(
                        f"任务信息共享池绑定不一致 ({rel_pf})：allowed_prefixes 与任务清单不一致"
                    )
            if "forbidden_prefixes" in data and linked_task.get("forbidden_prefixes") is not None:
                pool_forb = [normalize_rel_path(str(x)) for x in (data.get("forbidden_prefixes") or []) if normalize_rel_path(str(x))]
                if set(pool_forb) != set(linked_task["forbidden_prefixes"]):
                    errors.append(
                        f"任务信息共享池绑定不一致 ({rel_pf})：forbidden_prefixes 与任务清单不一致"
                    )

        if not isinstance(endpoints, dict) or not endpoints:
            errors.append(f"任务信息共享池分端状态缺失 ({rel_pf})：endpoints 必须是非空对象")
        else:
            for ep_name, ep_val in endpoints.items():
                if not isinstance(ep_val, dict) or not ep_val.get("role") or not ep_val.get("state"):
                    errors.append(
                        f"任务信息共享池分端条目无效 ({rel_pf} -> endpoints.{ep_name})：必须包含 role 与 state"
                    )

        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"任务信息共享池产物账本缺失 ({rel_pf})：artifacts 必须是非空列表")
            continue

        art_by_id: dict[str, dict[str, Any]] = {}
        for art in artifacts:
            if not isinstance(art, dict):
                errors.append(f"任务信息共享池产物条目格式错误 ({rel_pf})：条目必须是对象")
                continue
            art_id = str(art.get("artifact_id", "")).strip()
            art_name = str(art.get("name", "")).strip()
            holders = art.get("holders")
            if not art_id or not art_name or not isinstance(holders, dict) or not holders:
                errors.append(
                    f"任务信息共享池产物字段缺失 ({rel_pf} -> {art_id or '?'})：必须包含 artifact_id、name 与非空 holders"
                )
                continue
            art_by_id[art_id] = art

            if holders.get("github_repo") == "synced":
                req_files = art.get("required_repo_files_for_recalc") or []
                repo_path = normalize_rel_path(str(art.get("repo_path", "")).strip())
                if not req_files and not repo_path:
                    errors.append(
                        f"任务信息共享池假同步拦截 [{task_id}/{art_id}]：产物标记为 github_repo=synced，但未声明 repo_path 或 required_repo_files_for_recalc"
                    )
                elif repo_path.endswith("/") and not req_files:
                    errors.append(
                        f"任务信息共享池假同步拦截 [{task_id}/{art_id}]：目录产物 {repo_path} 标记为 github_repo=synced 时，必须列出 required_repo_files_for_recalc 逐个核验"
                    )
                for rf in req_files:
                    norm_rf = normalize_rel_path(str(rf))
                    in_head = git_run(repo, "cat-file", "-e", f"HEAD:{norm_rf}").returncode == 0
                    in_index = git_run(repo, "cat-file", "-e", f":{norm_rf}").returncode == 0
                    on_disk = (repo / norm_rf).is_file()
                    if not ((in_head or in_index) and on_disk):
                        errors.append(
                            f"任务信息共享池假同步拦截 [{task_id}/{art_id}]：产物标记为 github_repo=synced，但必需文件 {norm_rf} 未在 Git 仓库中跟踪入库（请检查是否因 .gitignore 漏执行 git add -f）"
                        )
                if repo_path:
                    in_head_rp = git_run(repo, "cat-file", "-e", f"HEAD:{repo_path}").returncode == 0
                    ls_rp = git_run(repo, "ls-files", repo_path)
                    has_tracked = ls_rp.returncode == 0 and bool(ls_rp.stdout.strip())
                    if not (in_head_rp or has_tracked) or not (repo / repo_path).exists():
                        errors.append(
                            f"任务信息共享池假同步拦截 [{task_id}/{art_id}]：产物标记为 github_repo=synced，但 repo_path {repo_path} 在 Git 仓库中不存在或未跟踪任何文件"
                        )

        if isinstance(endpoints, dict):
            for ep_name, ep_val in endpoints.items():
                if isinstance(ep_val, dict) and ep_val.get("state") == "blocked_on_unshared_artifact":
                    blocked_by = ep_val.get("blocked_by")
                    if not isinstance(blocked_by, list) or not blocked_by:
                        errors.append(
                            f"任务信息共享池阻塞声明不完整 ({rel_pf} -> {ep_name})：state=blocked_on_unshared_artifact 时必须提供非空 blocked_by 列表"
                        )
                    else:
                        for bid in blocked_by:
                            if bid not in art_by_id:
                                errors.append(
                                    f"任务信息共享池阻塞引用无效 ({rel_pf} -> {ep_name})：blocked_by 引用的产物 {bid} 不在 artifacts 中"
                                )

    return errors


def scan_commit(
    repo: Path,
    rev: str,
    git_errors: list[str],
    file_hashes: dict[str, str],
    expected_line: str | None = None,
    allowed_prefixes: Iterable[str] | None = None,
    allow_shared_index: bool = True,
    task_label: str | None = None,
    require_task_scope: bool = False,
    require_handoff_header: bool = False,
    require_public_ready: bool = False,
    forbidden_prefixes: Iterable[str] | None = None,
    active_tasks: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """扫描单个 commit 的路径归属/任务范围（含删除、重命名与合并提交）、公开发布门禁及现存改动文件的完整内容。"""
    parents_proc = git_checked(repo, git_errors, f"读取 commit {rev[:8]} 父提交", "rev-list", "--parents", "-n", "1", rev)
    if parents_proc is None:
        return [], [], {}
    parents = parents_proc.stdout.strip().split()[1:]
    if len(parents) >= 2:
        r = git_checked(
            repo,
            git_errors,
            f"枚举合并 commit {rev[:8]} 改动",
            "diff",
            "--name-status",
            "-M",
            f"{rev}^1",
            rev,
        )
    else:
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
    norm_own = [normalize_rel_path(p) for p in ownership_paths if normalize_rel_path(p)]
    norm_content = [normalize_rel_path(p) for p in content_paths if normalize_rel_path(p)]
    explicit_prefs = list(allowed_prefixes or [])
    explicit_forb = list(forbidden_prefixes or [])
    explicit_active_tasks = list(active_tasks or [])
    has_parent = git_run(repo, "rev-parse", "--verify", "-q", f"{rev}^").returncode == 0
    parent_has_binding = (
        has_parent
        and git_run(repo, "cat-file", "-e", f"{rev}^:{TASK_BINDING_REL_PATH}").returncode == 0
    )

    if norm_own == [TASK_BINDING_REL_PATH]:
        if TASK_BINDING_REL_PATH not in norm_content:
            if require_task_scope:
                git_errors.append(f"[{rev[:8]}] 未完成任务范围验收：禁止删除 {TASK_BINDING_REL_PATH}")
        else:
            bind_blob = git_run(repo, "show", f"{rev}:{TASK_BINDING_REL_PATH}")
            if bind_blob.returncode == 0 and bind_blob.stdout.strip():
                try:
                    parse_task_binding_text(bind_blob.stdout, strict_schema=True)
                except Exception as exc:
                    git_errors.append(f"[{rev[:8]}] 解析 {TASK_BINDING_REL_PATH} 失败：{exc}")
            else:
                git_errors.append(f"[{rev[:8]}] 读取 {TASK_BINDING_REL_PATH} 失败")

        if expected_line or explicit_prefs:
            ownership_issues, by_line = check_ownership(
                ownership_paths,
                expected_line=expected_line,
                allowed_prefixes=explicit_prefs,
                allow_shared_index=allow_shared_index,
                task_label=task_label,
                forbidden_prefixes=explicit_forb,
            )
        elif explicit_active_tasks:
            ownership_issues, by_line, matches = find_matching_tasks(
                ownership_paths, explicit_active_tasks
            )
            if matches:
                git_errors.extend(
                    f"[{rev[:8]}] {m}"
                    for m in _check_matched_tasks_public_ready(
                        matches, norm_own, require_public_ready
                    )
                )
        elif parent_has_binding:
            parent_tasks, parse_err = load_repo_active_tasks(
                repo, rev_spec=f"{rev}^", allow_disk_fallback_if_no_head=False
            )
            if parse_err and require_task_scope:
                git_errors.append(f"[{rev[:8]}] 解析父提交 {TASK_BINDING_REL_PATH} 失败：{parse_err}")
            if parent_tasks:
                ownership_issues, by_line, matches = find_matching_tasks(ownership_paths, parent_tasks)
                if matches:
                    git_errors.extend(
                        f"[{rev[:8]}] {m}"
                        for m in _check_matched_tasks_public_ready(
                            matches, norm_own, require_public_ready
                        )
                    )
            else:
                if require_task_scope and not parse_err:
                    git_errors.append(
                        f"[{rev[:8]}] 未完成任务范围验收：父提交 {TASK_BINDING_REL_PATH} 未包含有效活跃任务范围"
                    )
                ownership_issues, by_line = check_ownership(ownership_paths)
        else:
            ownership_issues, by_line = [], {"infra": [TASK_BINDING_REL_PATH]}
    elif expected_line or explicit_prefs:
        ownership_issues, by_line = check_ownership(
            ownership_paths,
            expected_line=expected_line,
            allowed_prefixes=explicit_prefs,
            allow_shared_index=allow_shared_index,
            task_label=task_label,
            forbidden_prefixes=explicit_forb,
        )
    elif explicit_active_tasks:
        ownership_issues, by_line, matches = find_matching_tasks(
            ownership_paths, explicit_active_tasks
        )
        if matches:
            git_errors.extend(
                f"[{rev[:8]}] {m}"
                for m in _check_matched_tasks_public_ready(
                    matches, norm_own, require_public_ready
                )
            )
    else:
        if not has_parent and set(norm_own) <= SHARED_INDEX_FILES:
            ownership_issues, by_line = check_ownership(ownership_paths)
        else:
            parent_tasks = []
            if has_parent:
                parent_tasks, parse_err = load_repo_active_tasks(
                    repo, rev_spec=f"{rev}^", allow_disk_fallback_if_no_head=False
                )
                if parse_err and require_task_scope:
                    git_errors.append(f"[{rev[:8]}] 解析父提交 {TASK_BINDING_REL_PATH} 失败：{parse_err}")
                elif not parent_tasks and require_task_scope:
                    git_errors.append(
                        f"[{rev[:8]}] 未完成任务范围验收：未提供任务绑定范围（父提交缺少有效的 {TASK_BINDING_REL_PATH}，且未传 --task-manifest / --allowed-prefixes / --expected-line）"
                    )
            else:
                if require_task_scope:
                    git_errors.append(
                        f"[{rev[:8]}] 未完成任务范围验收：根提交修改了非共用索引路径且缺少父提交 {TASK_BINDING_REL_PATH}"
                    )

            if parent_tasks:
                ownership_issues, by_line, matches = find_matching_tasks(ownership_paths, parent_tasks)
                if matches:
                    git_errors.extend(
                        f"[{rev[:8]}] {m}"
                        for m in _check_matched_tasks_public_ready(
                            matches, norm_own, require_public_ready
                        )
                    )
            else:
                ownership_issues, by_line = check_ownership(ownership_paths)

    ownership_issues = [f"[{rev[:8]}] {msg}" for msg in ownership_issues]

    privacy_findings: list[str] = []
    for rel_path in content_paths:
        blob = git_checked(repo, git_errors, f"读取 commit {rev[:8]} 文件 {rel_path}", "show", f"{rev}:{rel_path}")
        if blob is not None and "\x00" not in blob.stdout:
            label = f"{rev[:8]}:{rel_path}"
            sha = hashlib.sha256(blob.stdout.encode("utf-8")).hexdigest()
            file_hashes[label] = sha
            file_hashes[normalize_rel_path(rel_path)] = sha
            privacy_findings.extend(scan_text(label, blob.stdout))
            git_errors.extend(
                check_content_gates(
                    label,
                    blob.stdout,
                    require_handoff_header=require_handoff_header,
                    require_public_ready=require_public_ready,
                )
            )
    return ownership_issues, privacy_findings, by_line


def scan_uncommitted(
    repo: Path,
    git_errors: list[str],
    file_hashes: dict[str, str],
    expected_line: str | None = None,
    allowed_prefixes: Iterable[str] | None = None,
    allow_shared_index: bool = True,
    task_label: str | None = None,
    require_handoff_header: bool = False,
    require_public_ready: bool = False,
    active_tasks: list[dict[str, Any]] | None = None,
    forbidden_prefixes: Iterable[str] | None = None,
    require_task_scope: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """分别枚举并扫描暂存区（index）与工作区（worktree），返回 (ownership_issues, privacy_findings, scanned_labels)。"""
    ownership_issues: list[str] = []
    privacy_findings: list[str] = []
    scanned_labels: list[str] = []
    explicit_prefs = list(allowed_prefixes or [])
    explicit_forb = list(forbidden_prefixes or [])
    has_head = git_run(repo, "rev-parse", "--verify", "-q", "HEAD").returncode == 0
    head_has_binding = (
        has_head
        and git_run(repo, "cat-file", "-e", f"HEAD:{TASK_BINDING_REL_PATH}").returncode == 0
    )

    def _check_uncommitted_paths(paths_list: list[str], content_list: list[str], tag: str) -> list[str]:
        norm_p = [normalize_rel_path(p) for p in paths_list if normalize_rel_path(p)]
        norm_c = [normalize_rel_path(p) for p in content_list if normalize_rel_path(p)]
        if not norm_p:
            return []
        if norm_p == [TASK_BINDING_REL_PATH]:
            if TASK_BINDING_REL_PATH not in norm_c:
                if require_task_scope:
                    git_errors.append(f"[{tag}] 未完成任务范围验收：禁止删除 {TASK_BINDING_REL_PATH}")
            else:
                if tag == "index":
                    b = git_run(repo, "show", f":{TASK_BINDING_REL_PATH}")
                    raw_txt = b.stdout if b.returncode == 0 else ""
                else:
                    dp = repo / TASK_BINDING_REL_PATH
                    raw_txt = dp.read_text(encoding="utf-8", errors="replace") if dp.is_file() else ""
                try:
                    parse_task_binding_text(raw_txt, strict_schema=True)
                except Exception as exc:
                    git_errors.append(f"[{tag}] 解析 {TASK_BINDING_REL_PATH} 失败：{exc}")
            if expected_line or explicit_prefs:
                return check_ownership(
                    paths_list,
                    expected_line=expected_line,
                    allowed_prefixes=explicit_prefs,
                    allow_shared_index=allow_shared_index,
                    task_label=task_label,
                    forbidden_prefixes=explicit_forb,
                )[0]
            if active_tasks:
                issues, _, matches = find_matching_tasks(paths_list, active_tasks)
                if matches:
                    git_errors.extend(
                        f"[{tag}] {m}"
                        for m in _check_matched_tasks_public_ready(matches, norm_p, require_public_ready)
                    )
                return issues
            if head_has_binding:
                head_tasks, parse_err = load_repo_active_tasks(
                    repo, rev_spec="HEAD", allow_disk_fallback_if_no_head=False
                )
                if parse_err and require_task_scope:
                    git_errors.append(f"[{tag}] 解析 HEAD:{TASK_BINDING_REL_PATH} 失败：{parse_err}")
                if head_tasks:
                    issues, _, matches = find_matching_tasks(paths_list, head_tasks)
                    if matches:
                        git_errors.extend(
                            f"[{tag}] {m}"
                            for m in _check_matched_tasks_public_ready(matches, norm_p, require_public_ready)
                        )
                    return issues
                if require_task_scope and not parse_err:
                    git_errors.append(
                        f"[{tag}] 未完成任务范围验收：HEAD:{TASK_BINDING_REL_PATH} 未包含有效活跃任务范围"
                    )
                return check_ownership(paths_list)[0]
            return []
        if expected_line or explicit_prefs:
            return check_ownership(
                paths_list,
                expected_line=expected_line,
                allowed_prefixes=explicit_prefs,
                allow_shared_index=allow_shared_index,
                task_label=task_label,
                forbidden_prefixes=explicit_forb,
            )[0]
        if active_tasks:
            issues, _, matches = find_matching_tasks(paths_list, active_tasks)
            if matches:
                git_errors.extend(
                    f"[{tag}] {m}"
                    for m in _check_matched_tasks_public_ready(matches, norm_p, require_public_ready)
                )
            return issues
        return check_ownership(paths_list)[0]

    # 1. 暂存区（index vs HEAD）
    staged_res = git_checked(repo, git_errors, "枚举暂存区改动", "diff", "--cached", "--name-status", "-M")
    staged_own: list[str] = []
    staged_content: list[str] = []
    if staged_res is not None:
        staged_own, staged_content = parse_name_status(staged_res.stdout.splitlines())
        own_idx = _check_uncommitted_paths(staged_own, staged_content, "index")
        ownership_issues.extend(f"[index] {m}" for m in own_idx)
        for rel in staged_content:
            label = f"index:{rel}"
            scanned_labels.append(label)
            blob = git_checked(repo, git_errors, f"读取暂存区 blob {rel}", "show", f":{rel}")
            if blob is not None and "\x00" not in blob.stdout:
                sha = hashlib.sha256(blob.stdout.encode("utf-8")).hexdigest()
                file_hashes[label] = sha
                file_hashes[normalize_rel_path(rel)] = sha
                privacy_findings.extend(scan_text(label, blob.stdout))
                git_errors.extend(
                    check_content_gates(
                        label,
                        blob.stdout,
                        require_handoff_header=require_handoff_header,
                        require_public_ready=require_public_ready,
                    )
                )

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

    own_wt = _check_uncommitted_paths(wt_own, wt_content, "worktree")
    ownership_issues.extend(f"[worktree] {m}" for m in own_wt)

    if not own_idx_or_wt(ownership_issues) and (staged_own and wt_own):
        combined_paths = sorted(set(staged_own) | set(wt_own))
        combined_norm = [normalize_rel_path(p) for p in combined_paths if normalize_rel_path(p)]
        if combined_norm != [TASK_BINDING_REL_PATH]:
            if expected_line or explicit_prefs:
                own_combined, _ = check_ownership(
                    combined_paths,
                    expected_line=expected_line,
                    allowed_prefixes=explicit_prefs,
                    allow_shared_index=allow_shared_index,
                    task_label=task_label,
                    forbidden_prefixes=explicit_forb,
                )
            elif active_tasks:
                own_combined, _, matches = find_matching_tasks(combined_paths, active_tasks)
                if matches:
                    for m in _check_matched_tasks_public_ready(
                        matches, combined_norm, require_public_ready
                    ):
                        git_errors.append(f"[uncommitted] {m}")
            else:
                own_combined, _ = check_ownership(combined_paths)
            ownership_issues.extend(f"[uncommitted] {m}" for m in own_combined)

    all_uncommitted_norm = [normalize_rel_path(p) for p in sorted(set(staged_own) | set(wt_own)) if normalize_rel_path(p)]
    if (
        require_task_scope
        and all_uncommitted_norm
        and all_uncommitted_norm != [TASK_BINDING_REL_PATH]
        and not (expected_line or explicit_prefs or active_tasks)
    ):
        git_errors.append(
            f"[uncommitted] 未完成任务范围验收：未提供任务绑定范围（缺少 --task-manifest / --allowed-prefixes / --expected-line 或 HEAD 中有效的 {TASK_BINDING_REL_PATH}）"
        )

    for rel in wt_content:
        fp = repo / rel
        if fp.is_file():
            label = f"worktree:{rel}"
            scanned_labels.append(label)
            raw = fp.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            file_hashes[label] = sha
            file_hashes[normalize_rel_path(rel)] = sha
            if b"\x00" not in raw:
                text = raw.decode("utf-8", errors="replace")
                privacy_findings.extend(scan_text(label, text))
                git_errors.extend(
                    check_content_gates(
                        label,
                        text,
                        require_handoff_header=require_handoff_header,
                        require_public_ready=require_public_ready,
                    )
                )

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


def normalize_commit_range(rng: str) -> str:
    """将全零远端 SHA（如首次推送 0000...000..<sha>）规范化为完整可达历史 <sha>，不退化为单 commit。"""
    s = rng.strip()
    if ".." in s and "..." not in s:
        left, right = s.split("..", 1)
        if left and set(left) == {"0"} and len(left) >= 7 and right.strip():
            return right.strip()
    return s


def install_pre_push_hook(repo: Path) -> Path:
    """在本地 Git 仓库安装 pre-push 钩子（兼容标准仓库与 git worktree），确保公开发布前在本地拦截违规提交。"""
    if not (repo / ".git").exists():
        raise ValueError(f"无法解析 Git hooks 目录（请确认 {repo} 为有效 Git 仓库根目录）：缺少 .git")
    proc = git_run(repo, "rev-parse", "--git-path", "hooks/pre-push")
    if proc.returncode == 0 and proc.stdout.strip():
        raw_hook = Path(proc.stdout.strip())
        hook_path = raw_hook if raw_hook.is_absolute() else (repo / raw_hook).resolve()
    elif (repo / ".git").is_dir():
        hook_path = repo / ".git" / "hooks" / "pre-push"
    else:
        raise ValueError(f"无法解析 Git hooks 目录（请确认 {repo} 为有效 Git 仓库）：{proc.stderr.strip()}")
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_content = """#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
ZERO="0000000000000000000000000000000000000000"
while read -r local_ref local_sha remote_ref remote_sha; do
  if [ "$local_sha" = "$ZERO" ]; then
    continue
  fi
  if [ "$remote_sha" = "$ZERO" ]; then
    RANGE="${local_sha}"
  else
    RANGE="${remote_sha}..${local_sha}"
  fi
  python3 "$REPO_ROOT/agents/check_repo_hygiene.py" \\
    --repo "$REPO_ROOT" \\
    --commit-range "$RANGE" \\
    --ownership-mode error \\
    --require-task-scope \\
    --require-public-ready
done
"""
    hook_path.write_text(hook_content, encoding="utf-8")
    hook_path.chmod(0o755)
    return hook_path


def verify_bound_receipt(
    receipt_path: Path, current_file_hashes: dict[str, str]
) -> list[str]:
    """核对当前待审文件的实际字节 SHA-256 是否与已签发回执中绑定的版本完全一致（拒绝空输入）。"""
    if not current_file_hashes:
        return ["回执校验失败：待核对的文件哈希输入为空（必须读取待审文件的实际字节并计算 SHA-256 后再核对回执）"]
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

    for rel, bound_sha in bound_hashes.items():
        if rel not in current_file_hashes:
            errors.append(f"回执校验失败：原回执绑定的文件 {rel} 未提供当前实际字节哈希，无法完成版本核对")
        elif current_file_hashes[rel] != bound_sha:
            errors.append(
                f"版本失效：文件 {rel} 当前 SHA-256 ({current_file_hashes[rel][:12]}) 与回执绑定版本 ({bound_hashes[rel][:12]}) 不一致；"
                "旧版本的“通过”只适用于旧版本，新稿必须重新送检"
            )

    for rel in current_file_hashes:
        if rel not in bound_hashes:
            errors.append(f"版本失效：文件 {rel} 不在原回执绑定清单中，新稿须重新送检")
    return errors


_MD_LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _materialize_link_targets(
    md_text: str, logical_parent_in_repo: Path, repo_resolved: Path, tmp_repo: Path
) -> None:
    """将 Markdown 中引用且在原仓库确实存在的非 .md 目标建为占位（保留真实断链以供 check_archive.py 捕获）。"""
    for target in _MD_LINK_RE.findall(md_text):
        if re.match(r"(https?:|mailto:|#|[A-Za-z]:[/\\])", target):
            continue
        rel_target = urllib.parse.unquote(target.split("#")[0])
        if not rel_target:
            continue
        try:
            orig_dest = (logical_parent_in_repo / rel_target).resolve()
            rel_in_repo = orig_dest.relative_to(repo_resolved)
        except (ValueError, OSError):
            continue
        dst = tmp_repo / rel_in_repo
        if orig_dest.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif orig_dest.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")


def _build_archive_mirror_repo(
    repo: Path, file_mappings: list[tuple[Path, str]], tmp_repo: Path
) -> None:
    """在临时目录构造带真实 .md 树与已存在链接目标的 Git 镜像仓库，用于检验外部交接草稿拟合入效果。"""
    repo_resolved = repo.resolve()
    md_copies: list[tuple[str, Path]] = []
    for md_path in sorted(repo_resolved.rglob("*.md")):
        if ".git" in md_path.parts:
            continue
        try:
            rel_md = md_path.relative_to(repo_resolved)
        except ValueError:
            continue
        dst_md = tmp_repo / rel_md
        dst_md.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = md_path.read_bytes()
        dst_md.write_bytes(raw_bytes)
        md_copies.append((raw_bytes.decode("utf-8", errors="replace"), md_path.parent))

    for md_text, orig_parent in md_copies:
        _materialize_link_targets(md_text, orig_parent, repo_resolved, tmp_repo)

    subprocess.run(["git", "-C", str(tmp_repo), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_repo), "config", "user.name", "MirrorCheck"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_repo), "config", "user.email", "noreply@anthropic.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_repo), "commit", "-q", "-m", "mirror baseline", "--allow-empty"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(tmp_repo), "update-ref", "refs/remotes/origin/note", "HEAD"], check=True, capture_output=True)

    for fp, rel in file_mappings:
        norm_rel = normalize_rel_path(rel)
        dst = tmp_repo / norm_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = fp.read_bytes()
        dst.write_bytes(raw_bytes)
        logical_parent = (repo_resolved / norm_rel).parent
        _materialize_link_targets(raw_bytes.decode("utf-8", errors="replace"), logical_parent, repo_resolved, tmp_repo)


def run_archive_check_integration(
    archive_script: Path,
    archive_file: Path,
    repo: Path,
    skill_file: Path | None,
    file_mappings: list[tuple[Path, str]] | list[Path],
) -> dict[str, Any]:
    """联动调用私有学习线的 check_archive.py（支持平铺 Drive 存档 staging 与外部草稿仓库目标路径镜像），返回结构化回执记录。"""
    if not archive_script.is_file():
        return {
            "ran": False,
            "passed": False,
            "error": f"找不到 check_archive.py：{archive_script}",
        }
    if not archive_file.is_file():
        return {
            "ran": False,
            "passed": False,
            "error": f"找不到学习存档文件：{archive_file}",
        }

    repo_resolved = repo.resolve()
    normalized_mappings: list[tuple[Path, str]] = []
    for item in file_mappings:
        if isinstance(item, tuple):
            fp, rel = item[0].resolve(), normalize_rel_path(item[1])
        else:
            fp = Path(item).resolve()
            try:
                rel = normalize_rel_path(str(fp.relative_to(repo_resolved)))
            except ValueError:
                rel = fp.name
        normalized_mappings.append((fp, rel))

    has_skill_file = bool(skill_file and skill_file.is_file())
    need_stage_archive = not (archive_file.parent / "skill" / "SKILL.md").is_file() and has_skill_file
    need_mirror_repo = any(
        (not fp.is_relative_to(repo_resolved)) or (fp != (repo_resolved / rel).resolve())
        for fp, rel in normalized_mappings
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        arch_target = archive_file
        if need_stage_archive and skill_file is not None:
            arch_dir = tmp_root / "archive_stage"
            (arch_dir / "skill").mkdir(parents=True, exist_ok=True)
            arch_target = arch_dir / archive_file.name
            shutil.copy2(archive_file, arch_target)
            shutil.copy2(skill_file, arch_dir / "skill" / "SKILL.md")

        target_repo = repo_resolved
        if need_mirror_repo:
            target_repo = tmp_root / "repo_mirror"
            target_repo.mkdir(parents=True, exist_ok=True)
            _build_archive_mirror_repo(repo_resolved, normalized_mappings, target_repo)

        cmd = [sys.executable, str(archive_script), str(arch_target), "--repo", str(target_repo)]
        if has_skill_file and skill_file is not None:
            cmd.extend(["--skill", str(skill_file)])
        if normalized_mappings:
            cmd.extend(["--files", *[rel for _, rel in normalized_mappings]])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    sha = hashlib.sha256(archive_script.read_bytes()).hexdigest()
    arch_sha = hashlib.sha256(archive_file.read_bytes()).hexdigest()
    skill_sha = hashlib.sha256(skill_file.read_bytes()).hexdigest() if (has_skill_file and skill_file) else ""
    summary_line = ""
    for line in reversed((proc.stdout or "").splitlines()):
        if line.strip().startswith("结果："):
            summary_line = line.strip()
            break
    if proc.returncode != 0 and not summary_line:
        err_lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip().startswith("错误：")]
        if err_lines:
            summary_line = "；".join(err_lines[:3])
    return {
        "ran": True,
        "script": archive_script.name,
        "script_sha256": sha,
        "archive_file": archive_file.name,
        "archive_file_sha256": arch_sha,
        "skill_file": skill_file.name if (has_skill_file and skill_file) else "",
        "skill_file_sha256": skill_sha,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "summary": summary_line or (proc.stderr.strip()[:200] if proc.stderr else f"exit code {proc.returncode}"),
    }


def render_markdown_receipt(receipt: dict[str, Any]) -> str:
    """生成写回 Google Drive 任务目录的 Markdown 版本绑定检查回执。"""
    vis = receipt.get("visibility", "")
    st = receipt.get("status", "")
    if not receipt["passed"]:
        status_str = "未通过（FAILED）"
    elif vis in PUBLIC_VISIBILITY_VALUES and st in APPROVED_STATUS_VALUES:
        status_str = "公开发布前检查通过（PASSED，允许合入公开仓库）"
    else:
        status_str = "自动检查通过（仅代表格式/隐私/任务范围检查通过；内容待审，尚未获准公开发布）"

    lines = [
        "# 自动检查回执（绑定确定版本快照）",
        "",
        "> **版本绑定与状态说明**：本回执仅对下方表内列出的文件 `SHA-256` 快照生效，且**自动检查通过不代表学科内容已审核通过**。若写作者（如 Gemini 网页端）在回执生成后再次修改了 Drive 文件，文件内容哈希改变，本回执自动失效，新稿须重新送检。",
        "",
        "| 检查字段 | 记录值 |",
        "|---|---|",
        f"| 自动检查结论 | **{status_str}** |",
        f"| 卫生与范围检查器 | `{receipt['script']}` (`{receipt['script_sha256'][:12]}`) — `{'PASSED' if receipt['passed'] else 'FAILED'}` |",
    ]
    arch = receipt.get("archive_check")
    if arch and arch.get("ran"):
        lines.append(
            f"| 学习存档与笔记格式检查器 | `{arch['script']}` (`{arch['script_sha256'][:12]}`) — `{'PASSED' if arch['passed'] else 'FAILED'}`（{arch.get('summary', '')}） |"
        )
        if arch.get("archive_file_sha256"):
            skill_part = (
                f"`{arch['skill_file']}` (`{arch['skill_file_sha256'][:12]}`)"
                if arch.get("skill_file_sha256")
                else "未传入"
            )
            lines.append(
                f"| 绑定学习存档与 Skill | `{arch.get('archive_file', '')}` (`{arch['archive_file_sha256'][:12]}`) ｜ {skill_part} |"
            )
    else:
        lines.append("| 学习存档与笔记格式检查器 | 未在本回执联动运行 |")

    lines.extend([
        f"| 基线 Git Commit | `{receipt.get('git_head_commit') or 'N/A'}` |",
        f"| 绑定 Drive File ID | `{receipt.get('drive_file_id') or 'N/A'}` |",
        f"| 任务与负责人 | `{receipt.get('task_label') or 'N/A'}` |",
        f"| 允许修改范围 | `{', '.join(receipt.get('allowed_prefixes') or []) or '按工作线默认规则'}` |",
        f"| 可见范围与当前状态 | `{vis or 'N/A'}` ｜ `{st or 'N/A'}` |",
        "",
        "## 绑定文件快照（File SHA-256）",
        "",
    ])
    file_hashes: dict[str, str] = receipt.get("file_hashes", {})
    if file_hashes:
        lines.append("| 仓库目标路径 / 扫描标签 | 内容 SHA-256 |")
        lines.append("|---|---|")
        for rel, sha in sorted(file_hashes.items()):
            lines.append(f"| `{rel}` | `{sha}` |")
    else:
        lines.append("- 本次未扫描任何文件。")

    lines.extend([
        "",
        "## 检查明细",
        "",
        f"- 扫描 commit 数：`{len(receipt.get('scanned_commits', []))}`",
        f"- 扫描文件数：`{len(receipt.get('scanned_files', []))}`",
        f"- Git/元数据/门禁错误：`{len(receipt.get('git_errors', []))}`",
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
        "--forbidden-prefixes",
        nargs="*",
        help="限定当前任务禁止修改的路径前缀或精确文件列表",
    )
    ap.add_argument(
        "--task-manifest",
        help="读取任务范围与交接元数据文件（支持 JSON 或含 6 项元数据的 Markdown 交接文件）",
    )
    ap.add_argument(
        "--require-task-scope",
        action="store_true",
        help="强制要求本次检查具备任务级允许修改范围（从 CLI 参数、--task-manifest 或仓库父提交 agents/current_task.json 读取；缺失则报错）",
    )
    ap.add_argument(
        "--check-task-pool",
        action="store_true",
        help="校验仓库 agents/task_pools/*.json 任务级分端信息共享池格式及 github_repo=synced 产物完整性",
    )
    ap.add_argument(
        "--require-handoff-header",
        action="store_true",
        help="要求被扫描的交接文件顶部必须包含完整的 6 项交接必要元数据",
    )
    ap.add_argument(
        "--require-public-ready",
        action="store_true",
        help="公开发布前硬门禁：拦截可见范围为「工作区私有」或状态为「待审/退回/待提稿」的文件（适用于 --files、commit 扫描与暂存区/工作区扫描）",
    )
    ap.add_argument("--drive-file-id", help="记录在回执中的 Google Drive 文件 ID")
    ap.add_argument("--verify-receipt", help="核对当前待审文件的实际字节 SHA-256 是否仍与指定 JSON 回执绑定的版本一致")
    ap.add_argument("--archive-check-script", help="可选联动运行的 check_archive.py 脚本路径")
    ap.add_argument("--archive-file", help="联动运行 check_archive.py 时的学习存档路径")
    ap.add_argument("--skill-file", help="联动运行 check_archive.py 时的 SKILL.md 路径")
    ap.add_argument("--install-pre-push-hook", action="store_true", help="在仓库 .git/hooks/pre-push 安装发布前自动拦截钩子")
    ap.add_argument("--json-receipt", help="输出 JSON 验收回执的路径")
    ap.add_argument("--markdown-receipt", help="输出写回 Drive 的 Markdown 验收回执路径")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    if args.install_pre_push_hook:
        hook_p = install_pre_push_hook(repo)
        print(f"已安装发布前本地拦截钩子：{hook_p}")
        return 0
    git_errors: list[str] = []
    ownership_warnings: list[str] = []
    ownership_errors: list[str] = []
    privacy_errors: list[str] = []
    scanned_commits: list[str] = []
    scanned_files: list[str] = []
    file_hashes: dict[str, str] = {}
    archive_check_result: dict[str, Any] = {"ran": False}

    manifest_expected_line = args.expected_line
    expected_line = args.expected_line
    allowed_prefixes: list[str] = list(args.allowed_prefixes or [])
    forbidden_prefixes: list[str] = list(args.forbidden_prefixes or [])
    allow_shared_index = True
    task_label: str | None = None
    visibility: str = ""
    status: str = ""
    manifest_target_rel: str = ""
    manifest_active_tasks: list[dict[str, Any]] = []
    tm_resolved: Path | None = None
    explicit_scope_provided = bool(
        args.expected_line
        or args.allowed_prefixes
        or args.forbidden_prefixes
        or args.task_manifest
    )

    # 若指定了 --task-manifest，解析并合并任务范围与交接元数据
    if args.task_manifest:
        tm_path = Path(args.task_manifest)
        if not tm_path.is_file():
            git_errors.append(f"指定的任务清单文件不存在：{args.task_manifest}")
        else:
            tm_resolved = tm_path.resolve()
            try:
                manifest = parse_task_manifest(tm_path)
                if manifest.get("active_tasks"):
                    manifest_active_tasks = list(manifest["active_tasks"])
                else:
                    if args.require_handoff_header or args.require_public_ready:
                        git_errors.extend(validate_handoff_manifest(manifest, require_public_ready=args.require_public_ready))
                    elif manifest.get("visibility") or manifest.get("status"):
                        git_errors.extend(validate_handoff_manifest(manifest, require_public_ready=args.require_public_ready))
                    if not expected_line and manifest.get("expected_line"):
                        expected_line = manifest["expected_line"]
                        manifest_expected_line = expected_line
                    if not allowed_prefixes and manifest.get("allowed_prefixes"):
                        allowed_prefixes = list(manifest["allowed_prefixes"])
                    if not forbidden_prefixes and manifest.get("forbidden_prefixes"):
                        forbidden_prefixes = list(manifest["forbidden_prefixes"])
                    allow_shared_index = bool(manifest.get("allow_shared_index", True))
                    if manifest.get("task_id") or manifest.get("owner"):
                        task_label = " | ".join(x for x in (manifest.get("task_id"), manifest.get("owner")) if x)
                    visibility = manifest.get("visibility", "")
                    status = manifest.get("status", "")
                    manifest_target_rel = manifest.get("target_repo_path", "")
            except Exception as exc:
                git_errors.append(f"解析任务清单 {args.task_manifest} 失败：{exc}")

    effective_ownership_mode = (
        "error"
        if (
            args.ownership_mode == "error"
            or expected_line
            or allowed_prefixes
            or forbidden_prefixes
            or args.task_manifest
            or args.require_task_scope
            or args.check_task_pool
            or args.require_handoff_header
            or args.require_public_ready
        )
        else "warn"
    )

    resolved_file_paths: list[Path] = []
    file_mappings: list[tuple[Path, str]] = []
    matched_display_task: dict[str, Any] | None = None
    if args.files is not None:
        if not args.files:
            git_errors.append("--files 参数不能为空列表")
        rel_list: list[str] = []
        binding_disk_resolved = (repo / TASK_BINDING_REL_PATH).resolve()
        inline_expected_line: str | None = None
        inline_allowed_prefixes: list[str] = []
        inline_forbidden_prefixes: list[str] = []

        for idx, f in enumerate(args.files):
            fp = Path(f)
            if not fp.is_absolute():
                fp = repo / fp
            fp = fp.resolve()
            if not fp.is_file():
                git_errors.append(f"指定扫描的文件不存在：{f}")
                continue
            resolved_file_paths.append(fp)
            raw_bytes = fp.read_bytes()
            sha = hashlib.sha256(raw_bytes).hexdigest()
            text = raw_bytes.decode("utf-8", errors="replace")

            is_binding_target = fp == binding_disk_resolved or (
                idx == 0 and bool(args.target_rel_path) and normalize_rel_path(args.target_rel_path) == TASK_BINDING_REL_PATH
            )
            if not is_binding_target:
                try:
                    inline_manifest = parse_task_manifest_text(text, require_handoff_json_signature=True)
                except Exception:
                    inline_manifest = {}
            else:
                inline_manifest = {}

            inline_target_rel = ""
            if not inline_manifest.get("active_tasks"):
                if not inline_allowed_prefixes and inline_manifest.get("allowed_prefixes"):
                    inline_allowed_prefixes = list(inline_manifest["allowed_prefixes"])
                if not inline_forbidden_prefixes and inline_manifest.get("forbidden_prefixes"):
                    inline_forbidden_prefixes = list(inline_manifest["forbidden_prefixes"])
                if not inline_expected_line and inline_manifest.get("expected_line"):
                    inline_expected_line = inline_manifest["expected_line"]
                if not task_label and (inline_manifest.get("task_id") or inline_manifest.get("owner")):
                    task_label = " | ".join(
                        x for x in (inline_manifest.get("task_id"), inline_manifest.get("owner")) if x
                    )
                if not visibility and inline_manifest.get("visibility"):
                    visibility = inline_manifest.get("visibility", "")
                if not status and inline_manifest.get("status"):
                    status = inline_manifest.get("status", "")
                if inline_manifest.get("target_repo_path"):
                    inline_target_rel = inline_manifest["target_repo_path"]

            if args.target_rel_path and idx == 0:
                rel = normalize_rel_path(args.target_rel_path)
            elif manifest_target_rel and idx == 0:
                rel = normalize_rel_path(manifest_target_rel)
            elif inline_target_rel and idx == 0:
                rel = normalize_rel_path(inline_target_rel)
            else:
                try:
                    rel = normalize_rel_path(str(fp.relative_to(repo)))
                except ValueError:
                    rel = fp.name

            if args.require_handoff_header or args.require_public_ready:
                if tm_resolved is None or fp != tm_resolved or bool(manifest_active_tasks):
                    git_errors.extend(
                        check_content_gates(
                            rel,
                            text,
                            require_handoff_header=args.require_handoff_header,
                            require_public_ready=args.require_public_ready,
                        )
                    )

            rel_list.append(rel)
            file_mappings.append((fp, rel))
            scanned_files.append(rel)
            file_hashes[rel] = sha
            privacy_errors.extend(scan_text(rel, text))

        norm_rel_list = [normalize_rel_path(p) for p in rel_list if normalize_rel_path(p)]
        has_head = git_run(repo, "rev-parse", "--verify", "-q", "HEAD").returncode == 0
        head_has_binding = (
            has_head
            and git_run(repo, "cat-file", "-e", f"HEAD:{TASK_BINDING_REL_PATH}").returncode == 0
        )

        files_active_tasks = list(manifest_active_tasks)
        if not (expected_line or allowed_prefixes or files_active_tasks):
            loaded_tasks, parse_err = load_repo_active_tasks(
                repo, rev_spec="HEAD", allow_disk_fallback_if_no_head=True
            )
            if parse_err and args.require_task_scope:
                git_errors.append(f"解析 {TASK_BINDING_REL_PATH} 失败：{parse_err}")
            files_active_tasks = loaded_tasks
            if (
                not explicit_scope_provided
                and not files_active_tasks
                and not parse_err
                and not head_has_binding
                and norm_rel_list != [TASK_BINDING_REL_PATH]
                and not args.require_task_scope
                and (inline_expected_line or inline_allowed_prefixes)
            ):
                expected_line = inline_expected_line
                allowed_prefixes = list(inline_allowed_prefixes)
                forbidden_prefixes = list(inline_forbidden_prefixes)

        is_bootstrap_binding = norm_rel_list == [TASK_BINDING_REL_PATH] and not head_has_binding
        if (
            args.require_task_scope
            and not (expected_line or allowed_prefixes or files_active_tasks)
            and not is_bootstrap_binding
        ):
            git_errors.append(
                f"未完成任务范围验收：未提供任务绑定范围（缺少 --task-manifest / --allowed-prefixes / --expected-line 或有效的 {TASK_BINDING_REL_PATH}）"
            )

        if norm_rel_list == [TASK_BINDING_REL_PATH]:
            try:
                parse_task_binding_text(
                    resolved_file_paths[0].read_text(encoding="utf-8", errors="replace"),
                    strict_schema=True,
                )
            except Exception as exc:
                git_errors.append(f"解析 {TASK_BINDING_REL_PATH} 失败：{exc}")
            if expected_line or allowed_prefixes:
                own_issues, _ = check_ownership(
                    rel_list,
                    expected_line=expected_line,
                    allowed_prefixes=allowed_prefixes,
                    allow_shared_index=allow_shared_index,
                    task_label=task_label,
                    forbidden_prefixes=forbidden_prefixes,
                )
            elif files_active_tasks:
                own_issues, _, matches = find_matching_tasks(rel_list, files_active_tasks)
                if matches:
                    matched_display_task = next(
                        (t for t in matches if _task_public_ready_error(t) is None), matches[0]
                    )
                    git_errors.extend(
                        _check_matched_tasks_public_ready(matches, norm_rel_list, args.require_public_ready)
                    )
            elif head_has_binding:
                own_issues, _ = check_ownership(rel_list)
            else:
                own_issues = []
        elif expected_line or allowed_prefixes:
            own_issues, _ = check_ownership(
                rel_list,
                expected_line=expected_line,
                allowed_prefixes=allowed_prefixes,
                allow_shared_index=allow_shared_index,
                task_label=task_label,
                forbidden_prefixes=forbidden_prefixes,
            )
        elif files_active_tasks:
            own_issues, _, matches = find_matching_tasks(rel_list, files_active_tasks)
            if matches:
                matched_display_task = next(
                    (t for t in matches if _task_public_ready_error(t) is None), matches[0]
                )
                git_errors.extend(
                    _check_matched_tasks_public_ready(matches, norm_rel_list, args.require_public_ready)
                )
        else:
            own_issues, _ = check_ownership(rel_list)

        if matched_display_task is not None:
            if not task_label:
                task_label = " | ".join(
                    x for x in (matched_display_task.get("task_id"), matched_display_task.get("owner")) if x
                ) or None
            if not expected_line and matched_display_task.get("expected_line"):
                expected_line = matched_display_task["expected_line"]
            if not allowed_prefixes and matched_display_task.get("allowed_prefixes"):
                allowed_prefixes = list(matched_display_task["allowed_prefixes"])

        if effective_ownership_mode == "error":
            ownership_errors.extend(own_issues)
        else:
            ownership_warnings.extend(own_issues)
    else:
        if args.verify_receipt:
            git_errors.append("回执校验失败：使用 --verify-receipt 时必须同时通过 --files 指定待核对的文件以读取实际字节")

        chk = git_checked(repo, git_errors, "验证 Git 仓库", "rev-parse", "--git-dir")
        if chk is not None:
            explicit_range = args.commit_range is not None
            rng = args.commit_range if explicit_range else default_commit_range(repo)
            if rng:
                rng = normalize_commit_range(rng)
                revs_proc = git_checked(repo, git_errors, f"解析 commit 范围 {rng}", "rev-list", "--reverse", rng)
                if revs_proc is not None:
                    revs = [l.strip() for l in revs_proc.stdout.splitlines() if l.strip()]
                    if not revs and args.require_task_scope and not (expected_line or allowed_prefixes or manifest_active_tasks):
                        head_tasks, head_err = load_repo_active_tasks(
                            repo, rev_spec="HEAD", allow_disk_fallback_if_no_head=True
                        )
                        if head_err:
                            git_errors.append(f"解析 {TASK_BINDING_REL_PATH} 失败：{head_err}")
                        elif not head_tasks:
                            git_errors.append(
                                f"未完成任务范围验收：未提供任务绑定范围（缺少 --task-manifest / --allowed-prefixes / --expected-line 或有效的 {TASK_BINDING_REL_PATH}）"
                            )
                    for rev in revs:
                        scanned_commits.append(rev)
                        own_issues, priv_issues, _ = scan_commit(
                            repo,
                            rev,
                            git_errors,
                            file_hashes,
                            expected_line=expected_line,
                            allowed_prefixes=allowed_prefixes,
                            allow_shared_index=allow_shared_index,
                            task_label=task_label,
                            require_task_scope=args.require_task_scope,
                            require_handoff_header=args.require_handoff_header,
                            require_public_ready=args.require_public_ready,
                            forbidden_prefixes=forbidden_prefixes,
                            active_tasks=manifest_active_tasks,
                        )
                        if effective_ownership_mode == "error":
                            ownership_errors.extend(own_issues)
                        else:
                            ownership_warnings.extend(own_issues)
                        privacy_errors.extend(priv_issues)

            if args.include_worktree or (not explicit_range and not scanned_commits and not git_errors):
                wt_line = expected_line
                wt_prefs = list(allowed_prefixes)
                wt_forb = list(forbidden_prefixes)
                wt_shared = allow_shared_index
                wt_label = task_label
                wt_active_tasks = list(manifest_active_tasks)
                if not (wt_line or wt_prefs or wt_active_tasks):
                    loaded_wt_tasks, wt_err = load_repo_active_tasks(
                        repo, rev_spec="HEAD", allow_disk_fallback_if_no_head=True
                    )
                    if wt_err and args.require_task_scope:
                        git_errors.append(f"解析 HEAD:{TASK_BINDING_REL_PATH} 失败：{wt_err}")
                    wt_active_tasks = loaded_wt_tasks

                own_issues, priv_issues, labels = scan_uncommitted(
                    repo,
                    git_errors,
                    file_hashes,
                    expected_line=wt_line,
                    allowed_prefixes=wt_prefs,
                    allow_shared_index=wt_shared,
                    task_label=wt_label,
                    require_handoff_header=args.require_handoff_header,
                    require_public_ready=args.require_public_ready,
                    active_tasks=wt_active_tasks,
                    forbidden_prefixes=wt_forb,
                    require_task_scope=args.require_task_scope,
                )
                scanned_files.extend(labels)
                if effective_ownership_mode == "error":
                    ownership_errors.extend(own_issues)
                else:
                    ownership_warnings.extend(own_issues)
                privacy_errors.extend(priv_issues)

    if args.check_task_pool or args.require_task_scope:
        if manifest_active_tasks:
            pool_active_tasks = list(manifest_active_tasks)
        else:
            pool_active_tasks, pool_err = load_repo_active_tasks(
                repo, rev_spec="HEAD", allow_disk_fallback_if_no_head=True
            )
            if pool_err and args.check_task_pool and not args.require_task_scope:
                git_errors.append(f"解析 {TASK_BINDING_REL_PATH} 失败：{pool_err}")
        git_errors.extend(
            validate_task_pools(
                repo,
                active_tasks=pool_active_tasks,
                require_non_empty=bool(args.check_task_pool),
            )
        )

    if args.archive_check_script and args.archive_file:
        archive_check_result = run_archive_check_integration(
            Path(args.archive_check_script).resolve(),
            Path(args.archive_file).resolve(),
            repo,
            Path(args.skill_file).resolve() if args.skill_file else None,
            file_mappings,
        )
        if not archive_check_result.get("passed", False):
            git_errors.append(f"check_archive.py 检查未通过：{archive_check_result.get('summary') or archive_check_result.get('error')}")

    if args.verify_receipt and args.files is not None:
        git_errors.extend(verify_bound_receipt(Path(args.verify_receipt), file_hashes))

    total_err = len(git_errors) + len(ownership_errors) + len(privacy_errors)
    receipt = {
        "script": "agents/check_repo_hygiene.py",
        "script_sha256": script_sha256(),
        "git_head_commit": current_head_commit(repo),
        "drive_file_id": args.drive_file_id or "",
        "task_label": task_label or "",
        "expected_line": (manifest_expected_line or expected_line or ""),
        "allowed_prefixes": allowed_prefixes,
        "visibility": visibility,
        "status": status,
        "ownership_mode": effective_ownership_mode,
        "scanned_commits": scanned_commits,
        "scanned_files": scanned_files,
        "file_hashes": file_hashes,
        "archive_check": archive_check_result,
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
        print(f"错误（Git/元数据/门禁/回执）：{e}")
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
