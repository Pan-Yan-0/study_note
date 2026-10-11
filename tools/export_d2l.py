#!/usr/bin/env python3
"""把「D2L 伴学簿」的数据库导出（JSON）整理成 Markdown，写进仓库的 d2l 分支。

用法：
    python3 tools/export_d2l.py <db_root> <page_html> <out_dir>

- db_root：ArtifactData 导出时指定的 out_dir。里面应有 data/users/me/bk/{notes,subs,reviews}/<讲号>.json
  （也接受直接放着 notes/ subs/ reviews/ 三个目录）。
- page_html：伴学簿页面的 HTML（Artifact read 保存下来的那个文件）。从里面的 `const DATA = {...};`
  取出每一讲的标题、验收标准、回忆题和动手任务。
- out_dir：仓库根目录。脚本会重写 README.md 和 lectures/，不动别的文件。

只导出已提交的内容（submitted）、笔记（notes）和批改（reviews）；未提交的草稿不导出。
截图是页面的附件，不在数据库里，这里只记张数。
"""
import json
import re
import sys
import datetime
from pathlib import Path

STATUS_ORDER = {"pass": "通过", "rework": "需返工", "submitted": "已提交，待批改", "doing": "进行中", "todo": "未开始"}
TAG = {"must": "项目必需", "use": "有用", "ext": "拓展"}
HANDS = {"req": "必做", "rec": "建议做", "opt": "选做", "none": ""}


def load_data(page_html: Path):
    text = page_html.read_text(encoding="utf-8")
    i = text.index("const DATA = ") + len("const DATA = ")
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    return obj


def find_bk(root: Path) -> Path:
    for cand in (root / "data" / "users" / "me" / "bk", root):
        if (cand / "notes").is_dir() or (cand / "subs").is_dir() or (cand / "reviews").is_dir():
            return cand
    raise SystemExit(f"在 {root} 下找不到 notes/ subs/ reviews/")


def load_coll(bk: Path, name: str):
    d = bk / name
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def demote(md: str, by: int = 1) -> str:
    """把 Markdown 里代码块之外的标题降 by 级，避免和本文件自己的标题冲突。"""
    out, fence = [], False
    for line in (md or "").splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and re.match(r"^#{1,5}\s", line):
            line = "#" * by + line
        out.append(line)
    return "\n".join(out)


def blank_note(md: str) -> bool:
    """只有模板小标题、没有实际内容的笔记，算空。"""
    body = re.sub(r"^#{1,6}\s.*$", "", md or "", flags=re.M)
    return not body.strip()


def quote(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "> （空）"
    return "\n".join("> " + ln if ln.strip() else ">" for ln in text.splitlines())


def code_block(code: str, lang="python") -> str:
    code = (code or "").rstrip()
    fence = "````" if "```" in code else "```"
    return f"{fence}{lang}\n{code}\n{fence}"


def status_of(sub, rev):
    if rev:
        return "pass" if rev.get("verdict") == "pass" else "rework"
    if sub and sub.get("submitted"):
        return "submitted"
    return "doing"


def fmt_time(iso):
    if not iso:
        return ""
    try:
        t = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return t.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso


def render_review(rev, lec):
    out = []
    verdict = "通过" if rev.get("verdict") == "pass" else "需返工"
    out.append(f"**结论：{verdict}（第 {rev.get('round')} 轮）** · {fmt_time(rev.get('reviewedAt'))}")
    out.append("")
    acc = lec.get("acc", [])
    for n, it in enumerate(rev.get("items", [])):
        mark = "✓" if it.get("ok") else "✗"
        a = acc[n] if n < len(acc) else ""
        out.append(f"{n + 1}. {mark} {a}")
        if it.get("note"):
            out.append(f"   - {it['note'].strip().replace(chr(10), chr(10) + '     ')}")
    out.append("")
    if rev.get("comments"):
        out.append("**评语**")
        out.append("")
        out.append(demote(rev["comments"], 3))
    hist = rev.get("history") or []
    if hist:
        out.append("")
        out.append("<details><summary>此前各轮的批改</summary>")
        out.append("")
        for h in hist:
            v = "通过" if h.get("verdict") == "pass" else "需返工"
            out.append(f"#### 第 {h.get('round')} 轮：{v}（{fmt_time(h.get('reviewedAt'))}）")
            out.append("")
            out.append(demote(h.get("comments", ""), 3))
            out.append("")
        out.append("</details>")
    return "\n".join(out)


def render_lecture(lec, note, sub, rev):
    st = status_of(sub, rev)
    meta = [f"{TAG.get(lec['tag'], '')}"]
    if lec.get("task") and HANDS.get(lec.get("hands")):
        meta.append(f"动手{HANDS[lec['hands']]}")
    out = [f"# {lec['id']} {lec['t']}", "",
           f"- 状态：{STATUS_ORDER[st]}" + (f"（第 {rev['round']} 轮）" if rev else ""),
           f"- 属于：{lec['part']} · " + " · ".join(m for m in meta if m), ""]
    out += ["## 验收标准", ""]
    out += [f"{n + 1}. {a}" for n, a in enumerate(lec.get("acc", []))]
    out += [""]
    if note and not blank_note(note.get("md", "")):
        out += ["## 笔记", "", demote(note["md"], 1).strip(), ""]
    s = (sub or {}).get("submitted")
    if s:
        out += [f"## 提交（第 {s.get('round')} 轮 · {fmt_time(s.get('at'))}）", ""]
        qs = lec.get("qs", [])
        if qs:
            out += ["### 回忆题", ""]
            ans = s.get("answers") or []
            for n, q in enumerate(qs):
                out += [f"**{n + 1}. {q}**", "", quote(ans[n] if n < len(ans) else ""), ""]
        if lec.get("task"):
            out += ["### 动手任务", "", f"任务：{lec['task']}", ""]
            for label, key in (("改了什么", "changed"), ("运行前的预测", "predicted"), ("实际结果", "actual"), ("报错与原因", "errors")):
                if (s.get(key) or "").strip():
                    out += [f"**{label}**", "", quote(s[key]) if key != "actual" else code_block(s[key], "text"), ""]
            if (s.get("code") or "").strip():
                out += ["**代码**", "", code_block(s["code"]), ""]
            if (s.get("link") or "").strip():
                out += [f"**链接**：{s['link'].strip()}", ""]
        shots = s.get("shots") or []
        if shots:
            out += [f"（另有截图 {len(shots)} 张，是页面的附件，没有导出。）", ""]
    if rev:
        out += ["## 批改", "", render_review(rev, lec), ""]
    return "\n".join(out).rstrip() + "\n"


def safe(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", name).strip("_")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    db_root, page_html, out_dir = map(Path, sys.argv[1:])
    data = load_data(page_html)
    bk = find_bk(db_root)
    notes, subs, revs = load_coll(bk, "notes"), load_coll(bk, "subs"), load_coll(bk, "reviews")
    lectures = {l["id"]: l for l in data["lectures"]}

    lec_dir = out_dir / "lectures"
    lec_dir.mkdir(parents=True, exist_ok=True)
    for old in lec_dir.glob("*.md"):
        old.unlink()

    files, status = {}, {}
    for lid, lec in lectures.items():
        note, sub, rev = notes.get(lid), subs.get(lid), revs.get(lid)
        has_note = bool(note and not blank_note(note.get("md", "")))
        has_sub = bool(sub and sub.get("submitted"))
        if not (has_note or has_sub or rev):
            status[lid] = "todo"
            continue
        status[lid] = status_of(sub, rev)
        fname = f"{lid}_{safe(lec['t'])}.md"
        (lec_dir / fname).write_text(render_lecture(lec, note, sub, rev), encoding="utf-8")
        files[lid] = fname

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    n_pass = sum(1 for s in status.values() if s == "pass")
    lines = [
        "# 动手学深度学习（李沐，v2）学习记录", "",
        "每讲的笔记、回忆题回答、动手记录和批改。内容来自「D2L 伴学簿」，由 Claude 每周日 10:00（北京时间）整理推送到这个分支；批改通过的讲，笔记和提交已经过修正。", "",
        f"- 最后更新：{now}（北京时间）",
        f"- 进度：通过 {n_pass} / {len(lectures)} 讲",
        "- 练习项目：SemEval-2027 Task 2（DiCo-NLI），四个轨道", "",
        "状态说明：通过 / 需返工（批改后等用户修改）/ 已提交待批改 / 进行中（只有笔记）/ 未开始。", "",
    ]
    for b in data["blocks"]:
        lines += [f"## 第 {b['no']} 组 {b['name']}", "", "| 讲 | 标题 | 状态 |", "|---|---|---|"]
        for lid in b["ids"]:
            lec = lectures[lid]
            title = f"[{lec['t']}](lectures/{files[lid]})" if lid in files else lec["t"]
            lines.append(f"| {lid} | {title} | {STATUS_ORDER[status[lid]]} |")
        lines.append("")
    lines += ["## 目录说明", "",
              "- `lectures/`：每讲一个文件，包含验收标准、笔记、提交和批改；", "- `tools/export_d2l.py`：生成上面这些文件的脚本。", ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"导出 {len(files)} 讲；通过 {n_pass}；README 与 lectures/ 已写入 {out_dir}")


if __name__ == "__main__":
    main()
