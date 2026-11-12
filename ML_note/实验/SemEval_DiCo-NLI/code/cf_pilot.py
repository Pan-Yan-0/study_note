"""反事实最小对（方向 B）的试点：选父样本和对照、生成 Haiku 批次、格式检查、盲审表、计分。
冻结计划变更记录 11-12「研究对齐 v4」条目 B 部分；规则在生成和审核之前写定，看结果后不改。

  python3 -I code/cf_pilot.py build  <官方仓库目录> <输出目录>             选父样本和对照，写 batches/*.json 和 pilot_key.json（封存）
  python3 -I code/cf_pilot.py prep   <输出目录> <运行目录>                 写 prompts/<batch>.txt（逐字节等于批次文件的 prompt）
  python3 -I code/cf_pilot.py harvest <输出目录> <运行目录> <subagents 目录> 从 transcript 登记每次派发（同 haiku_dispatch 的核对）
  python3 -I code/cf_pilot.py sheet  <输出目录> <运行目录>                 合并合格编辑，抽审核样本，写 review_sheet.csv 和 review_key.json
  python3 -I code/cf_pilot.py score  <输出目录> <复审后的标注.csv>          计分，写 review_result.json

父样本（只用本折 fit 的金标和文本，不用任何模型预测）
  fold 0 的 fit 家族里，金标 E 的家族；取 pair_id 最小的 __original 实例作代表。两段都是 1–6 个词，至少共享一个实词
  （"共享同一中心概念"的文本近似，编辑提示词再要求编辑保持中心词），两段不同。
  排除 Claude 看过金标的家族：boundary_review.py 的 seen_by_claude（单词替换盲审、E 类对照和参考卡、E 复审、P1、规模复审的
  答案文件，打印过的顽固错误）加上边界盲审 review_key.json 的 48 个家族。
  种子 SEED 抽 60 个家族：30 个做 CF-SPEC（改 text1 → 候选 F、改 text2 → 候选 B，各 15），30 个做 CF-HEAD（改哪一侧交替，候选 O）。
  每种编辑分 3 批、每批 10 项。

两种编辑（提示词见 PROMPTS）
  CF-SPEC：在指定一侧插入一个连续的 1–3 词修饰语，其余词不变、不改顺序；修饰语要真正缩小外延，不能是另一侧已经说出或蕴含的内容，
    不能改时态或真实性（former、ex-、fake 之类），不能是数字或数量。
  CF-HEAD：把指定一侧的中心词换成同类、非同义、非上下位、非部分整体的另一个词，其余词不变（a / an 可随之调整），
    新词不能出现在另一侧，不能是数字。
  做不到时返回 skip 和理由，不硬写。

格式检查（逐项；整批只在 JSON 结构不对时算 format_error）
  token = 字母数字串（可带撇号后缀）或单个标点，比较时转小写。
  CF-SPEC：新短语 = 原短语在某个位置插入一个连续的 1–3 个 token，其中至少一个含字母，且都不含数字。
  CF-HEAD：和原短语等长，恰好一个位置不同（另允许一处 a ↔ an）；新词全是字母、和原词不同、不是原词的词形变化
    （一方是另一方的前缀且长度差 ≤ 3）、不出现在另一侧，且不含数字。
  不合格的项过滤掉并计数，不重派；skip 单独计数。

盲审（Claude 初标 → Codex 复审，按复审结果计分；审阅人只看两个短语和编号，对照参考卡
  metrics/augment_b/e_recheck/参考卡.md 判 E / F / B / O / U，交表后才揭示候选标签和对照金标）
  每种编辑从合格项里抽 20 对（不足 20 就全取）。
  同现象的原数据对照，各 20 对，排除看过金标的家族和 60 个父样本家族，每个家族最多一行：
    SPEC 对照：金标 F 且 text1 = text2 插入 1–3 个 token，或金标 B 且 text2 = text1 插入 1–3 个 token（F、B 各 10）；
    HEAD 对照：金标 O 且两段等长、恰好一个 token 不同。
  展示顺序随机（种子 SEED）。

事先定的放行规则（score 输出）
  每种编辑分别计：分歧率 = 复审标签 ≠ 候选标签（CF）或 ≠ 金标（对照）的对数 / 可判定对数（非 U）；U 的数量单独报。
  CF 分歧率 ≤ 同现象对照的分歧率 + 15 个百分点（整数交叉相乘判定，不用浮点）则该编辑放行。两种编辑分开判定。
  另报：每种编辑的生成数、合格数、skip 数、各项过滤原因；CF-SPEC 改 text1 / text2 分开的分歧；审阅人给 E 的 CF-SPEC 对数
  （冗余修饰语）；CF-HEAD 被判成 F / B 的对数（新词其实是上下位或部分整体）。
"""
import csv, glob, hashlib, io, json, os, random, re, sys
from collections import Counter

SEED = 20261113
SE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SE}/code")
sys.path.insert(0, f"{SE}/code/augment_b")
SHORT = {"E": "EQUIVALENCE", "F": "FORWARD_ENTAILMENT", "B": "BACKWARD_ENTAILMENT", "O": "NEGATIVE_OTHER", "U": "UNSURE"}
LONG = {v: k for k, v in SHORT.items()}
N_PER_OP, N_REVIEW, BATCH = 30, 20, 10
TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\sA-Za-z0-9]")

PROMPTS = {
"SPEC": """You are editing short English phrases for a phrase-level natural language inference dataset. The phrases are chunks like those in image captions or news headlines.

Each item below gives two phrases, P1 and P2, that mean the same thing, and says which one to edit.

WHAT TO DO
Insert ONE restrictive modifier into the phrase you are told to edit, so that the edited phrase becomes strictly more specific than before: it still talks about the same thing, but now covers only some of the situations the original covered.

RULES
1. Insert 1 to 3 consecutive words: an adjective, a short adjective phrase, or a short prepositional phrase. Do not change, remove or reorder any existing word.
2. Keep the head word, its meaning and the roles of the people or things in the scene.
3. The modifier must really narrow the meaning. Do not add something redundant ("wet water"), a pure intensifier ("very"), or a detail that the other phrase already states or implies.
4. Do not change time or truth: no "former", "ex-", "future", "fake", "alleged" and the like. Do not add numbers or quantities.
5. Keep the style of the phrase: same use of capital letters and articles.
6. If no such edit is possible for an item, skip it and give a short reason instead of forcing an edit.
7. Use only the information in this message. Do not open files, search the web or look at any other material.

ITEMS
{items}

OUTPUT
Return only one JSON object and nothing else, with no Markdown code fence:
{{"batch_id": "{batch_id}", "edits": [{{"k": 1, "new_phrase": "..."}}, {{"k": 2, "skip": "reason"}}, ...]}}
Give exactly one entry for each k from 1 to {n}, in order. "new_phrase" is the whole edited phrase.
""",
"HEAD": """You are editing short English phrases for a phrase-level natural language inference dataset. The phrases are chunks like those in image captions or news headlines.

Each item below gives two phrases, P1 and P2, that mean the same thing, and says which one to edit.

WHAT TO DO
In the phrase you are told to edit, replace its head word (the main thing the phrase is about; for a verb phrase, the main verb) with a different word of the same kind, so that afterwards neither phrase follows from the other. Examples of good replacements: dog -> cat, guitar -> violin, running -> swimming.

RULES
1. Replace exactly one word. Keep every other word unchanged and in the same order; you may only switch "a" and "an" if the new word needs it.
2. The new word must be of the same kind as the old one, but it must NOT be a synonym, a broader term ("animal" for "dog"), a narrower term ("poodle" for "dog"), a part or a whole of it, or just another form of the same word.
3. The new word must not appear in the other phrase, and must not be a number.
4. Keep the grammar correct (same singular or plural, same verb form).
5. If no such edit is possible for an item, skip it and give a short reason instead of forcing an edit.
6. Use only the information in this message. Do not open files, search the web or look at any other material.

ITEMS
{items}

OUTPUT
Return only one JSON object and nothing else, with no Markdown code fence:
{{"batch_id": "{batch_id}", "edits": [{{"k": 1, "new_phrase": "..."}}, {{"k": 2, "skip": "reason"}}, ...]}}
Give exactly one entry for each k from 1 to {n}, in order. "new_phrase" is the whole edited phrase.
"""}


def sha(b): return hashlib.sha256(b).hexdigest()
def rj(p): return json.load(open(p, encoding="utf-8"))
def wj(p, x):
    with open(p, "w", encoding="utf-8") as f: json.dump(x, f, ensure_ascii=False, indent=1)
def toks(t): return [x.lower() for x in TOKEN.findall(t)]


def insertion(orig, new):
    """new = orig 插入一个连续片段时返回插入的 token 列表，否则 None。"""
    a, b = toks(orig), toks(new)
    k = len(b) - len(a)
    if k <= 0: return None
    for i in range(len(a) + 1):
        if b[:i] == a[:i] and b[i + k:] == a[i:]: return b[i:i + k]
    return None


def substitution(orig, new):
    """等长、恰好一个位置不同（另允许一处 a ↔ an）时返回 (旧词, 新词)，否则 None。"""
    a, b = toks(orig), toks(new)
    if len(a) != len(b): return None
    d = [i for i in range(len(a)) if a[i] != b[i]]
    art = [i for i in d if {a[i], b[i]} == {"a", "an"}]
    d = [i for i in d if i not in art]
    if len(d) != 1 or len(art) > 1: return None
    return a[d[0]], b[d[0]]


def check_item(op, orig, other, new):
    if not isinstance(new, str) or not new.strip(): return "空"
    if any(c.isdigit() for c in new) and not any(c.isdigit() for c in orig): return "加了数字"
    if new.strip().lower() == other.strip().lower(): return "和另一侧相同"
    if op == "SPEC":
        ins = insertion(orig, new)
        if ins is None: return "不是单处插入"
        if not 1 <= len(ins) <= 3: return f"插入 {len(ins)} 个 token"
        if not any(re.search(r"[a-z]", t) for t in ins): return "插入的只有标点"
        if any(re.search(r"\d", t) for t in ins): return "插入了数字"
        return None
    s = substitution(orig, new)
    if s is None: return "不是单词替换"
    old, w = s
    if not w.isalpha(): return "新词不是纯字母"
    if w == old: return "新旧相同"
    if (w.startswith(old) or old.startswith(w)) and abs(len(w) - len(old)) <= 3: return "词形变化"
    if w in toks(other): return "新词在另一侧出现"
    return None


def check(b, text):
    """haiku_dispatch.harvest 用的整批检查：只有 JSON 结构不对才算 format_error；逐项结果另存。"""
    out = {"raw_sha256": sha(text.encode("utf-8")), "reasons": []}
    try:
        x = json.loads(text)
        assert isinstance(x, dict) and x.get("batch_id") == b["batch_id"], "batch_id 不对"
        ed = x.get("edits")
        assert isinstance(ed, list) and [e.get("k") for e in ed] == list(range(1, len(b["items"]) + 1)), "k 不是 1..n"
        for e in ed:
            assert ("new_phrase" in e) != ("skip" in e), f"第 {e.get('k')} 项要么给 new_phrase 要么给 skip"
    except Exception as e:  # noqa: BLE001
        out.update(status="format_error", reasons=[f"{type(e).__name__}: {e}"]); return out
    out["status"] = "ok"
    return out


def seen_families(off):
    import boundary_review as br
    g = br.load_analysis(off)
    pairs, n_files, n_printed = br.seen_by_claude(g)
    byid, gold, fam = g["byid"], g["gold"], g["fam"]
    family = {i: fam["family_of_pair"][byid[i]["pair_id"]] for i in gold}
    seen = {family[i] for i in gold if byid[i]["pair_id"] in pairs}
    bkey = rj(f"{SE}/metrics/diagnostics/boundary_review/review_key.json")
    seen |= {v["family"] for v in bkey["items"].values()}
    return seen, {"answer_files": n_files, "printed": n_printed, "boundary_review_families": len(bkey["items"])}


def build(off, out):
    rows = list(csv.DictReader(open(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8")))
    byid = {r["instance_id"]: r for r in rows}
    fam = rj(f"{SE}/metrics/diagnostics/families.json")
    family = {r["instance_id"]: fam["family_of_pair"][r["pair_id"]] for r in rows}
    fold = {i: int(fam["fold_of_family"][str(family[i])]) for i in byid}
    seen, seen_src = seen_families(off)
    import boundary_review as br
    content = lambda t: {x for x in toks(t) if x.isalpha() and x not in br.STOP}
    rep = {}
    for r in rows:
        i = r["instance_id"]
        if i.endswith("__original") and (family[i] not in rep or r["pair_id"] < byid[rep[family[i]]]["pair_id"]): rep[family[i]] = i
    cand = []
    for f, i in sorted(rep.items(), key=lambda kv: kv[1]):
        r = byid[i]
        if r["label"] != SHORT["E"] or fold[i] == 0 or f in seen: continue
        n1, n2 = len(r["text1"].split()), len(r["text2"].split())
        if not (1 <= n1 <= 6 and 1 <= n2 <= 6) or r["text1"].strip().lower() == r["text2"].strip().lower(): continue
        if not content(r["text1"]) & content(r["text2"]): continue
        cand.append(i)
    rng = random.Random(SEED)
    pick = rng.sample(cand, 2 * N_PER_OP)
    spec, head = pick[:N_PER_OP], pick[N_PER_OP:]
    items = []
    for n, i in enumerate(spec):
        side = "text1" if n % 2 == 0 else "text2"
        items.append({"op": "SPEC", "parent": i, "edit_side": side, "target": SHORT["F"] if side == "text1" else SHORT["B"]})
    for n, i in enumerate(head):
        items.append({"op": "HEAD", "parent": i, "edit_side": "text1" if n % 2 == 0 else "text2", "target": SHORT["O"]})
    os.makedirs(f"{out}/batches", exist_ok=True)
    batches = {}
    for op in ("SPEC", "HEAD"):
        its = [x for x in items if x["op"] == op]
        for bn in range(0, len(its), BATCH):
            bid = f"cf_{op.lower()}_b{bn // BATCH + 1:02d}"
            chunk = its[bn:bn + BATCH]
            lines = []
            for k, x in enumerate(chunk, 1):
                r = byid[x["parent"]]
                lines.append(f'{k}. P1: "{r["text1"]}" | P2: "{r["text2"]}" | edit: {"P1" if x["edit_side"] == "text1" else "P2"}')
            prompt = PROMPTS[op].format(items="\n".join(lines), batch_id=bid, n=len(chunk))
            b = {"batch_id": bid, "op": op, "prompt": prompt, "prompt_sha256": sha(prompt.encode("utf-8")),
                 "items": [{"k": k, "parent": x["parent"], "edit_side": x["edit_side"], "orig": byid[x["parent"]][x["edit_side"]],
                            "other": byid[x["parent"]]["text2" if x["edit_side"] == "text1" else "text1"]} for k, x in enumerate(chunk, 1)]}
            wj(f"{out}/batches/{bid}.json", b); batches[bid] = b
    parents = {family[x["parent"]] for x in items}
    # 同现象的原数据对照（每个家族一行）
    spec_f, spec_b, head_c = [], [], []
    for f in sorted(set(family.values())):
        if f in seen or f in parents: continue
        for i in sorted(i for i in byid if family[i] == f):
            r = byid[i]; l = r["label"]
            ins_1 = insertion(r["text2"], r["text1"]); ins_2 = insertion(r["text1"], r["text2"])
            if l == SHORT["F"] and ins_1 and len(ins_1) <= 3: spec_f.append(i); break
            if l == SHORT["B"] and ins_2 and len(ins_2) <= 3: spec_b.append(i); break
            if l == SHORT["O"] and substitution(r["text1"], r["text2"]): head_c.append(i); break
    # 一个家族同时有 F 方向和 B 方向时，上面只取了 pair_id 排序在前的那一行；F、B 两个池可能来自同一家族的不同方向，不会重复取同一家族
    ctrl = {"SPEC": rng.sample(spec_f, 10) + rng.sample(spec_b, 10), "HEAD": rng.sample(head_c, 20)}
    key = {"version": "cf-pilot-v1", "seed": SEED, "rules": "见 code/cf_pilot.py 开头",
           "candidates": len(cand), "excluded_seen_families": len(seen), "seen_sources": seen_src,
           "items": items, "controls": {op: [{"instance_id": i, "family": family[i], "gold": byid[i]["label"]} for i in v] for op, v in ctrl.items()},
           "control_pools": {"SPEC_F": len(spec_f), "SPEC_B": len(spec_b), "HEAD": len(head_c)},
           "batches": {k: v["prompt_sha256"] for k, v in batches.items()}}
    wj(f"{out}/pilot_key.json", key)
    print(f"候选 E 家族 {len(cand)}（排除看过金标的 {len(seen)} 个）；SPEC {len(spec)}、HEAD {len(head)}；批次 {len(batches)}；"
          f"对照池 SPEC-F {len(spec_f)} / SPEC-B {len(spec_b)} / HEAD {len(head_c)}")


def prep(out, run):
    os.makedirs(f"{run}/prompts", exist_ok=True)
    for p in sorted(glob.glob(f"{out}/batches/*.json")):
        b = rj(p)
        path = f"{run}/prompts/{b['batch_id']}.txt"
        with open(path, "w", encoding="utf-8", newline="") as f: f.write(b["prompt"])
        assert sha(open(path, "rb").read()) == b["prompt_sha256"]
    import haiku_dispatch as hd
    print(f"{len(os.listdir(f'{run}/prompts'))} 个提示词文件；派发说明 {hd.WRAPPER_SHA[:12]}…")


def harvest(out, run, sub):
    import haiku_dispatch as hd
    hd.check = check
    hd.harvest(out, run, sub)


def collect(out, run):
    """每批取最后一次 ok 的尝试，逐项检查。"""
    log = [json.loads(l) for l in open(f"{run}/dispatch_log.jsonl", encoding="utf-8")]
    res, stats = [], {"SPEC": Counter(), "HEAD": Counter()}
    for p in sorted(glob.glob(f"{out}/batches/*.json")):
        b = rj(p)
        ok = [e for e in log if e["batch_id"] == b["batch_id"] and e["status"] == "ok"]
        assert ok, f"{b['batch_id']} 没有合格的尝试"
        e = ok[-1]
        raw = open(f"{run}/outputs/{b['batch_id']}/attempt_{e['attempt']}.txt", encoding="utf-8").read()
        assert sha(raw.encode("utf-8")) == e["raw_sha256"]
        ed = json.loads(raw)["edits"]
        for it, x in zip(b["items"], ed):
            st = stats[b["op"]]; st["generated"] += 1
            if "skip" in x: st["skip"] += 1; continue
            why = check_item(b["op"], it["orig"], it["other"], x["new_phrase"])
            if why: st["filtered:" + why] += 1; continue
            st["valid"] += 1
            res.append({"batch_id": b["batch_id"], "k": it["k"], "op": b["op"], "parent": it["parent"], "edit_side": it["edit_side"],
                        "new_phrase": x["new_phrase"].strip(), "other": it["other"]})
    return res, {k: dict(v) for k, v in stats.items()}


def sheet(out, run, off=None):
    key = rj(f"{out}/pilot_key.json")
    target = {(x["parent"], x["op"]): x["target"] for x in key["items"]}
    res, stats = collect(out, run)
    rng = random.Random(SEED + 1)
    rows = []
    for op in ("SPEC", "HEAD"):
        v = [x for x in res if x["op"] == op]
        for x in (rng.sample(v, N_REVIEW) if len(v) > N_REVIEW else v):
            t1, t2 = (x["new_phrase"], x["other"]) if x["edit_side"] == "text1" else (x["other"], x["new_phrase"])
            rows.append({"kind": "CF", "op": op, "text1": t1, "text2": t2, "expected": target[(x["parent"], op)],
                         "parent": x["parent"], "edit_side": x["edit_side"], "batch_id": x["batch_id"], "k": x["k"]})
    off = off or os.environ.get("DICO_OFFICIAL")
    assert off, "需要官方仓库目录（第三个参数或 DICO_OFFICIAL）"
    byid = {r["instance_id"]: r for r in csv.DictReader(open(f"{off}/final_data/train/dico_nli_train_track1_participant_labeled.csv", encoding="utf-8"))}
    for op, v in key["controls"].items():
        for c in v:
            r = byid[c["instance_id"]]
            rows.append({"kind": "CTRL", "op": op, "text1": r["text1"], "text2": r["text2"], "expected": c["gold"], "instance_id": c["instance_id"]})
    rng.shuffle(rows)
    with open(f"{out}/review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["review_id", "text1", "text2", "label", "evidence"])
        for n, r in enumerate(rows, 1): w.writerow([f"C{n:03d}", r["text1"], r["text2"], "", ""])
    rk = {"version": "cf-pilot-review-v1", "seed": SEED + 1, "generation_stats": stats, "valid_pool": len(res),
          "items": {f"C{n:03d}": r for n, r in enumerate(rows, 1)}}
    wj(f"{out}/review_key.json", rk)
    wj(f"{out}/valid_edits.json", res)
    print("生成统计", stats, "| 审核表", len(rows), "行", Counter((r["kind"], r["op"]) for r in rows))
    print("review_sheet.csv", sha(open(f"{out}/review_sheet.csv", "rb").read())[:12], "review_key.json",
          sha(open(f"{out}/review_key.json", "rb").read())[:12])


def score(out, labels):
    rk = rj(f"{out}/review_key.json")
    lab = {r["review_id"]: r["label"].strip().upper() for r in csv.DictReader(open(labels, encoding="utf-8"))}
    assert set(lab) == set(rk["items"]) and all(v in SHORT for v in lab.values()), "标注表不完整或有非法标签"
    res = {"rule": "每种编辑：CF 分歧率 ≤ 同现象对照分歧率 + 15 个百分点（整数交叉相乘）", "ops": {}}
    for op in ("SPEC", "HEAD"):
        cell = {}
        for kind in ("CF", "CTRL"):
            its = [(rid, x) for rid, x in rk["items"].items() if x["op"] == op and x["kind"] == kind]
            dec = [(rid, x) for rid, x in its if lab[rid] != "U"]
            dis = [rid for rid, x in dec if SHORT[lab[rid]] != x["expected"]]
            cell[kind] = {"n": len(its), "U": len(its) - len(dec), "decidable": len(dec), "disagree": len(dis),
                          "labels": dict(Counter(lab[rid] for rid, _ in its))}
        a, b = cell["CF"], cell["CTRL"]
        # a.dis / a.dec ≤ b.dis / b.dec + 15/100  ⇔  100·a.dis·b.dec ≤ 100·b.dis·a.dec + 15·a.dec·b.dec
        ok = a["decidable"] > 0 and b["decidable"] > 0 and 100 * a["disagree"] * b["decidable"] <= 100 * b["disagree"] * a["decidable"] + 15 * a["decidable"] * b["decidable"]
        cell["pass"] = ok
        if op == "SPEC":
            for side in ("text1", "text2"):
                its = [rid for rid, x in rk["items"].items() if x["op"] == op and x["kind"] == "CF" and x["edit_side"] == side and lab[rid] != "U"]
                cell[f"CF_{side}"] = {"decidable": len(its), "disagree": sum(SHORT[lab[r]] != rk["items"][r]["expected"] for r in its)}
            cell["CF_judged_E"] = sum(1 for rid, x in rk["items"].items() if x["op"] == op and x["kind"] == "CF" and lab[rid] == "E")
        else:
            cell["CF_judged_F_or_B"] = sum(1 for rid, x in rk["items"].items() if x["op"] == op and x["kind"] == "CF" and lab[rid] in "FB")
        res["ops"][op] = cell
    res["generation_stats"] = rk["generation_stats"]
    res["labels_sha256"] = sha(open(labels, "rb").read())
    wj(f"{out}/review_result.json", res)
    for op, c in res["ops"].items():
        print(op, "pass" if c["pass"] else "fail", "CF", c["CF"]["disagree"], "/", c["CF"]["decidable"], "| 对照", c["CTRL"]["disagree"], "/", c["CTRL"]["decidable"])


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2: raise SystemExit(__doc__)
    if a[1] == "build": build(os.path.abspath(a[2]), os.path.abspath(a[3]))
    elif a[1] == "prep": prep(os.path.abspath(a[2]), os.path.abspath(a[3]))
    elif a[1] == "harvest": harvest(os.path.abspath(a[2]), os.path.abspath(a[3]), os.path.abspath(a[4]))
    elif a[1] == "sheet": sheet(os.path.abspath(a[2]), os.path.abspath(a[3]), os.path.abspath(a[4]) if len(a) > 4 else None)
    elif a[1] == "score": score(os.path.abspath(a[2]), os.path.abspath(a[3]))
    else: raise SystemExit(__doc__)
