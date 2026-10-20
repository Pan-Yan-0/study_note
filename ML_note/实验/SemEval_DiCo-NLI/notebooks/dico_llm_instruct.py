# %% [markdown]
# # DiCo-NLI 第三组实验：Qwen 指令提示（零样本、少样本、CoT；不训练）
#
# 只有两个代码单元：先运行配置单元，再运行第二个单元（定义 + 推理 + 评分 + 打包），也可以直接 Run All。
#
# 对应导师清单第 1 项"大模型 instruct、Qwen"和第 2 项里的 CoT。不更新模型权重，只靠指令让 `Qwen/Qwen3-4B-Instruct-2507` 判断两个短语的关系。这个模型只有非思考模式；`cot` 是让它把推理写出来的提示，不是打开模型自带的思考模式。
#
# **四种提示词**（都用导师 SST-2 材料里的 Alpaca 格式，放进模型自己的对话模板；选项 A、B、C、D 依次对应 EQUIVALENCE、FORWARD、BACKWARD、NEGATIVE_OTHER，D 写作"以上都不是"）：
#
# - `zero`：零样本，只答一个字母。
# - `zero_rules`：在 `zero` 上只多一段"从训练数据归纳的启发式"：没有 at most、exactly 这类词的普通计数，数大的推出数小的；读数不同则互不蕴含。规则是看过训练标签后归纳的（见 README 的盲审一节），报告时称"训练标签归纳规则提示"，它用到了任务的监督信息。
# - `few`：在 `zero` 上只多 8 个训练集例子（每类 2 个）。例子按固定种子从英语训练集里抽，四个赛道用同一批 pair（各用本赛道的文本）。避开和 dev 文本相同的短语对，也不用盲审过的 45 个家族：它们是从模型出错的格子里抽的，其中 11 个的金标两位审阅者都不认同。
# - `cot`：在 `zero` 上改成先写推理、最后一行写 "Answer: X"，开头加导师材料里那句 "Before answering, think carefully ..."。最多生成 320 个 token，没写到答案的也在这里强制接上 "Answer:" 读概率，所以它是"最多 320 token 推理后强制作答"。
#
# **怎么取答案**：不解析自由文本。在模型回答的开头替它写上 "Answer:"，读下一个 token 是 A、B、C、D 的概率：每个字母的候选 token（"A" 和 " A"）先做 logsumexp，再在四类之间 softmax。`cot` 先生成推理，保留 "Answer:" 之前的全部推理，再统一接上 "\nAnswer:" 读。四个字母合起来占的概率叫 answer mass，记均值、低分位数和最低的几条例子，用来检查模型是不是按要求作答。它只说明模型愿意在这个位置写字母，不等于自由回答的格式遵循率。
#
# **两种预测**：`plain` 只看 (text1, text2)；`sym` 再把两句对调问一次，把对调结果的 F / B 概率互换后和原方向取平均，和编码器实验的对称化相同。相同的有序输入只算一次。
#
# **评测顺序和选择规则（事先定好）**：
#
# 1. 先跑英语训练集（`zero`、`zero_rules`、`few`）。评估集去掉少样本例子所在的家族，再分两部分分别评分：
#    - `select`：再去掉计数规则的 26 个来源家族。三种提示词按这里对称化后的 WF1 选一个（相同时取靠前的），写进 `selection.json`。
#    - `rule_source`：这 26 个来源家族。`zero_rules` 在这里的分数是循环的（规则就是从它们归纳的），只作参考。
# 2. 再跑官方 dev 四个赛道，四种提示词都跑，只报告，不改上面的选择。`cot` 不在训练集上跑，是事先声明的独立对照，不按 dev 成绩把它挑出来。
#
# **和导师 SST-2 材料的区别**：任务是四选一；答案读字母的概率，不解析回答文本，所以没有"无效回答"，改为报告 answer mass；模型在本机用 transformers 加载，不经过 Ollama 或 API；`cot` 用贪心解码，不采样。
#
# **断点续跑**：每算完一个 GPU 批次就写进缓存。缓存最后一行如果只写了一半（上次被打断），会先备份、再截掉这半行，补算对应的输入；中间的行损坏则停下。每个阶段跑完都更新一次结果包 `dico_llm_results_<本次开始时间>.zip`，出错退出时也尽量打包。换平台时把结果包放到工作目录（Kaggle 上作为数据集挂到 `/kaggle/input`）：只从设置相同的包里恢复，有多个时选进度最多的。模型版本、提示词、数据和代码版本必须和原来一致，否则会停下并提示。
#
# **批大小**：默认保守，读概率每批 4,096 个 token，生成每批 8 条；显存不够时自动减半重试。SMOKE 会按正式长度（320 个 token）测一次生成和最长的少样本输入，打印峰值显存和建议的批大小。改批大小只影响速度，不影响续跑；但 fp16 下批的组成不同，极接近的概率可能有微小差别。
#
# **时间估计（T4，未实测）**：dev 四个赛道共 6,776 个有序输入，英语训练集 3,450 个。按默认批大小，`zero`、`zero_rules`、`few` 合计约 2 小时，`cot` 约 2.5–3 小时（每批 8 条时生成较慢；按 SMOKE 的建议调大 `GEN_BATCH` 会快很多）。按 4B 模型、T4 约 10 TFLOPS 估算，每个阶段跑完会打印实际用时。
#
# **SMOKE = True**：只跑赛道 1 和 4、每个文件取 24 条，`cot` 最多生成 64 个 token。另外检查三件事：几条训练集上的原始回答；同一输入单独评分和放进长短混合的批里评分，概率差多少；正式长度下的显存和用时。结果也存进 `smoke_checks.json`。先跑 SMOKE，把输出发给 Claude 看过再正式跑。
#
# 在 Kaggle 上要打开 Internet（克隆官方仓库、下载约 8 GB 的模型）。Colab、腾讯 Cloud Studio 直接运行；Cloud Studio 上如果下载不了 Hugging Face 的模型，先在第一个单元里加一行 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`。

# %%
import os

SMOKE = False
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROMPTS = ["zero", "zero_rules", "few", "cot"]
TRACKS = [1, 2, 3, 4]
TRAIN_PROMPTS = ["zero", "zero_rules", "few"]  # 先在英语训练集上跑，用来做选择
FEW_PER_LABEL = 2
EXAMPLE_SEED = 2027
COT_MAX_NEW_TOKENS = 320
TOKENS_PER_BATCH = 4096  # 读概率时一批最多多少个 token；SMOKE 会给出建议值，显存不够时自动减半
GEN_BATCH = 8  # cot 生成时一批几条；同上
SMOKE_ROWS = 24
SMOKE_COT_TOKENS = 64

WORK = os.path.abspath(os.environ.get("DICO_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."))
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
OUT = os.path.join(WORK, "dico_llm_smoke" if SMOKE else "dico_llm_results")  # SMOKE 的结果单独放，不会被当成正式结果续跑

if SMOKE:
    TRACKS = [1, 4]

# %%
import csv
import glob
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter

VERSION = "dico_llm_instruct-2"
TRANSFORMERS_VERSION = "5.19.0"  # 和 IMDB 的 Qwen3-4B 实验相同
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LETTERS = ["A", "B", "C", "D"]
SWAP = [0, 2, 1, 3]  # 对调两句后：FORWARD ↔ BACKWARD，其余不变
RUN_STAMP = time.strftime("%Y%m%d-%H%M")
SCORE_CHUNK = 32  # 读概率时每算这么多条写一次缓存

HEADER = ("Below is an instruction that describes a task, paired with an input that provides "
          "further context. Write a response that appropriately completes the request.")
THINK_STEP = ("Before answering, think carefully about the question and create a step-by-step "
              "chain of thoughts to ensure a logical and accurate response.")
TASK = ("You are given two short phrases taken from image captions or news headlines. The phrases may be "
        "in English, Spanish or Basque. Decide which inference relation holds between Phrase 1 and Phrase 2.\n"
        "A. Equivalence: each phrase entails the other.\n"
        "B. Forward entailment: Phrase 1 entails Phrase 2, but Phrase 2 does not entail Phrase 1.\n"
        "C. Backward entailment: Phrase 2 entails Phrase 1, but Phrase 1 does not entail Phrase 2.\n"
        "D. Other: none of A, B or C applies.")
RULES = ("Use the following dataset-derived heuristics when applicable:\n"
         "- For plain counts of people or things, with no other quantity word such as \"at most\" or \"exactly\", "
         "the phrase with the larger count entails the phrase with the smaller count (\"7 boats\" entails \"3 boats\").\n"
         "- Phrases that differ only in a measured value, such as a magnitude, rate, percentage or date, do not "
         "entail each other.")
ANSWER_LETTER = "Answer with a single letter: A, B, C or D. No further explanation is required."
ANSWER_COT = ("Write your reasoning first, then give the final answer on the last line in the form "
              "\"Answer: X\", where X is A, B, C or D.")
MARKER = "Answer:"
SELECTION_RULE = ("在英语训练集的 select 子集（去掉少样本例子家族和计数规则的 26 个来源家族）上，"
                  "zero、zero_rules、few 中对称化后 WF1 最高的；相同时取靠前的")
# 盲审过的 45 个家族（metrics/diagnostics/blind_review/）：从模型出错的格子里抽的，其中 11 个的金标两位审阅者都不认同，不拿来当例子
REVIEWED_FAMILIES = {f"dico_phrasis_{i}" for i in (
    "0000079", "0000202", "0000267", "0000420", "0000473", "0000646", "0000670", "0000673", "0000682", "0000709",
    "0000853", "0000854", "0000919", "0000969", "0001024", "0001054", "0001158", "0001161", "0001187", "0001238",
    "0001401", "0001438", "0001566", "0001720", "0001774", "0001917", "0001942", "0001997", "0002082", "0002225",
    "0002275", "0002410", "0002420", "0002509", "0002511", "0002557", "0002578", "0002820", "0002833", "0002861",
    "0002998", "0003042", "0003079", "0003092", "0003316")}
# 计数规则的 26 个来源 pair（metrics/analysis/label_conventions.json）：训练集上比较提示词时，排除它们所在的家族
RULE_SOURCE_PAIRS = {f"dico_phrasis_{i}" for i in (
    "0000009", "0000076", "0000170", "0000267", "0000376", "0000403", "0000416", "0000420", "0000422", "0001338",
    "0001401", "0001541", "0001717", "0001736", "0001737", "0002422", "0002461", "0002472", "0002502", "0002517",
    "0002718", "0002730", "0002743", "0002771", "0002773", "0002963")}


def build_prompt(kind, a, b, examples=()):
    head = HEADER + ("\n" + THINK_STEP if kind == "cot" else "")
    instruction = "### Instruction:\n" + TASK
    if kind == "zero_rules":
        instruction += "\n" + RULES
    instruction += "\n" + (ANSWER_COT if kind == "cot" else ANSWER_LETTER)
    parts = [head, instruction]
    if kind == "few":
        parts.append("### Examples:\n" + "\n\n".join(
            f"Phrase 1: {e['text1']}\nPhrase 2: {e['text2']}\n{MARKER} {e['letter']}" for e in examples))
    parts.append(f"### Input:\nPhrase 1: {a}\nPhrase 2: {b}")
    parts.append("### Response:")
    return "\n\n".join(parts)


def cot_tokens():
    return SMOKE_COT_TOKENS if SMOKE else COT_MAX_NEW_TOKENS


# ---------- 环境、数据 ----------

def ensure_packages():
    try:
        have = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        have = None
    if have == TRANSFORMERS_VERSION:
        return
    if "transformers" in sys.modules:
        raise SystemExit(f"这个内核里已经加载了 transformers {have}，需要 {TRANSFORMERS_VERSION}。"
                         "请重启内核（Restart），再从头运行。")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_VERSION}"], check=True)
    importlib.invalidate_caches()


def ensure_official_repo():
    if not os.path.isdir(os.path.join(DICO_DIR, "final_data")):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git", DICO_DIR], check=True)
    print("官方仓库：", DICO_DIR)


def data_file(track, split, kind):
    return os.path.join(DICO_DIR, "final_data", split, f"dico_nli_{split}_track{track}_{kind}.csv")


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    return " ".join(s.lower().split())


def pair_key(r):
    return tuple(sorted((norm(r["text1"]), norm(r["text2"]))))


def build_families(rows):
    """规范化后文本相同（不计两侧顺序）的 pair_id 合成一个家族，和第二组 notebook 相同。"""
    parent, first = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        p = r["pair_id"]
        parent.setdefault(p, p)
        k = pair_key(r)
        if k in first:
            a, b = find(p), find(first[k])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[k] = p
    return {p: find(p) for p in parent}


def train_families():
    return build_families(read_rows(data_file(1, "train", "participant_labeled")))


def pick_examples():
    """少样本例子：每类 FEW_PER_LABEL 个 pair，取自英语训练集；避开和 dev 文本相同的短语对和盲审过的家族，每个家族最多用一次。"""
    train, dev = read_rows(data_file(1, "train", "participant_labeled")), read_rows(data_file(1, "dev", "participant_labeled"))
    dev_keys = {pair_key(r) for r in dev}
    fam = build_families(train)
    by_pair = {}
    for r in train:
        by_pair.setdefault(r["pair_id"], []).append(r)
    pools = {l: [] for l in LABELS}
    for pid in sorted(by_pair):
        if fam[pid] in REVIEWED_FAMILIES or any(pair_key(r) in dev_keys for r in by_pair[pid]):
            continue
        for label in {r["label"] for r in by_pair[pid]}:
            pools[label].append(pid)
    rng = random.Random(EXAMPLE_SEED)
    chosen, used = [], set()
    for label in LABELS:
        cands = pools[label][:]
        rng.shuffle(cands)
        n = 0
        for p in cands:
            if n == FEW_PER_LABEL:
                break
            if fam[p] not in used:
                used.add(fam[p])
                chosen.append({"pair_id": p, "label": label})
                n += 1
    rng.shuffle(chosen)
    assert len(chosen) == FEW_PER_LABEL * 4 and len({fam[c["pair_id"]] for c in chosen}) == len(chosen)
    assert Counter(c["label"] for c in chosen) == Counter({l: FEW_PER_LABEL for l in LABELS})
    return chosen, used


def examples_for_track(track, chosen):
    """每个例子在本赛道里找同一 pair、同一标签的行；赛道 4 有 6 种语言组合，按种子固定选一种。"""
    rows = read_rows(data_file(track, "train", "participant_labeled"))
    out = []
    for c in chosen:
        cands = sorted((r for r in rows if r["pair_id"] == c["pair_id"] and r["label"] == c["label"]),
                       key=lambda r: r["instance_id"])
        if not cands:
            raise ValueError(f"赛道 {track} 里找不到例子 {c}")
        r = random.Random(f"{EXAMPLE_SEED}-{c['pair_id']}").choice(cands)
        out.append({"instance_id": r["instance_id"], "text1": r["text1"], "text2": r["text2"],
                    "letter": LETTERS[LABELS.index(c["label"])]})
    return out


def stage_rows(split, track, excluded):
    rows = read_rows(data_file(track, split, "participant_labeled"))
    if split == "train":
        fam = build_families(rows)
        rows = [r for r in rows if fam[r["pair_id"]] not in excluded]
    return rows[:SMOKE_ROWS] if SMOKE else rows


def train_subsets(rows):
    """训练集评估分两部分：select 去掉计数规则的来源家族，用来选提示词；rule_source 是这些来源家族。"""
    fam = train_families()
    rule = {fam[p] for p in RULE_SOURCE_PAIRS}
    return {"select": [r for r in rows if fam[r["pair_id"]] not in rule],
            "rule_source": [r for r in rows if fam[r["pair_id"]] in rule]}


def needed_inputs(rows):
    need, seen = [], set()
    for r in rows:
        for x in ((r["text1"], r["text2"]), (r["text2"], r["text1"])):
            if x not in seen:
                seen.add(x)
                need.append(x)
    return need


def key_of(a, b):
    return json.dumps([a, b], ensure_ascii=False)


def load_jsonl(path):
    """读缓存。最后一行只写了一半时（上次被打断），备份后截掉它；中间的行损坏则停下。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    pos = good_end = 0
    for k, raw in enumerate(lines):
        end = pos + len(raw) + (1 if k < len(lines) - 1 else 0)
        if raw.strip():
            try:
                d = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if any(x.strip() for x in lines[k + 1:]):
                    raise SystemExit(f"{path} 第 {k + 1} 行损坏，而且不在末尾。请检查这个文件后再运行。")
                backup = f"{path}.broken-{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.copyfile(path, backup)
                with open(path, "r+b") as f:
                    f.truncate(good_end)
                print(f"  {os.path.basename(path)} 最后一行不完整（上次被打断），已备份到 {backup} 并截掉，这些输入会重新计算")
                return out
            out[key_of(d["a"], d["b"])] = d
            good_end = end
        pos = end
    if data and not data.endswith(b"\n"):  # 最后一条完整但没写换行，补上，免得下一条接在同一行
        with open(path, "ab") as f:
            f.write(b"\n")
    return out


def append_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for d in items:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


# ---------- 指纹、结果包 ----------

def fingerprint(chosen):
    files = sorted({data_file(t, s, k) for t in (1, 2, 3, 4) for s in ("train", "dev")
                    for k in ("participant_labeled", "reference")})
    return {
        "version": VERSION, "model": MODEL, "transformers": TRANSFORMERS_VERSION, "dtype": "float16",
        "prompt_text": {"header": HEADER, "think_step": THINK_STEP, "task": TASK, "rules": RULES,
                        "answer_letter": ANSWER_LETTER, "answer_cot": ANSWER_COT, "marker": MARKER},
        "examples": chosen, "few_per_label": FEW_PER_LABEL, "example_seed": EXAMPLE_SEED,
        "reviewed_families_excluded": sorted(REVIEWED_FAMILIES), "rule_source_pairs": sorted(RULE_SOURCE_PAIRS),
        "selection_rule": SELECTION_RULE, "cot_max_new_tokens": cot_tokens(), "decoding": "greedy",
        "smoke": SMOKE, "smoke_rows": SMOKE_ROWS if SMOKE else None,
        "data_sha256": {os.path.relpath(p, DICO_DIR): sha256(p) for p in files},
    }


def incompatible(old, fp):
    return sorted(k for k in set(fp) | set(old) if k != "model_revision" and fp.get(k) != old.get(k))


def restore_previous_results(fp):
    """只从设置相同的结果包恢复；有多个时选完成阶段最多、其次缓存条数最多的。"""
    if SMOKE or os.path.isdir(OUT):
        return
    base = os.path.basename(OUT)
    found = sorted(set(glob.glob(os.path.join(WORK, f"{base}_*.zip")) + glob.glob(f"/kaggle/input/*/{base}_*.zip")
                       + glob.glob(f"/kaggle/input/*/*/{base}_*.zip")))
    best, best_progress = None, None
    for path in found:
        try:
            with zipfile.ZipFile(path) as z:
                old = json.loads(z.read(f"{base}/fingerprint.json"))
                if incompatible(old, fp):
                    print("  跳过设置不同的结果包：", path)
                    continue
                names = z.namelist()
                progress = (sum(n.endswith("done.json") for n in names),
                            sum(z.read(n).count(b"\n") for n in names if n.endswith(".jsonl")))
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            print("  跳过读不了的结果包：", path)
            continue
        if best_progress is None or progress > best_progress:
            best, best_progress = path, progress
    if best:
        zipfile.ZipFile(best).extractall(WORK)
        print(f"已从结果包恢复（完成 {best_progress[0]} 个阶段，缓存 {best_progress[1]} 条）：", best)


def check_fingerprint(fp):
    path = os.path.join(OUT, "fingerprint.json")
    if os.path.exists(path):
        old = read_json(path)
        diff = incompatible(old, fp)
        if diff:
            raise SystemExit(f"当前设置和已有结果不一致：{diff}。要接着跑，请把设置改回去；要重新开始，请删除或改名 {OUT}。")
        return old.get("model_revision")
    write_json(path, fp)
    return None


def record_revision(revision):
    path = os.path.join(OUT, "fingerprint.json")
    fp = read_json(path)
    if fp.get("model_revision") not in (None, revision):
        raise SystemExit(f"模型版本和上次不同：{fp['model_revision']} → {revision}")
    fp["model_revision"] = revision
    write_json(path, fp)


def code_hash():
    """几个决定结果的函数的源码哈希，只记录、不参与续跑检查（改了这些函数要手动提高 VERSION）。"""
    try:
        src = "".join(inspect.getsource(f) for f in (build_prompt, cot_continuation, describe_generation, sym,
                                                      run_stage, QwenBackend))
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return None


def record_run(backend):
    path = os.path.join(OUT, "run_meta.json")
    runs = read_json(path) if os.path.exists(path) else []
    runs.append({"start": RUN_STAMP, "python": platform.python_version(), "tokens_per_batch": TOKENS_PER_BATCH,
                 "gen_batch": GEN_BATCH, "code_sha256": code_hash(), **backend.info()})
    write_json(path, runs)


def pack():
    if not os.path.isdir(OUT):
        return None
    zip_path = os.path.join(WORK, f"{os.path.basename(OUT)}_{RUN_STAMP}.zip")
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
            if os.path.isfile(path) and not path.endswith(".tmp"):
                z.write(path, os.path.relpath(path, WORK))
    os.replace(tmp, zip_path)
    return zip_path


# ---------- 模型（唯一用到 torch 的部分） ----------

class QwenBackend:
    def __init__(self, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if not torch.cuda.is_available():
            raise SystemExit("没有检测到 GPU。请在 Kaggle / Colab 里打开 GPU（T4）再运行。")
        self.torch, self.GenerationConfig = torch, GenerationConfig
        self.tokens_per_batch, self.gen_batch = TOKENS_PER_BATCH, GEN_BATCH
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=revision, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.option_ids = []
        for letter in LETTERS:
            ids = sorted({t[0] for form in (letter, " " + letter)
                          for t in [self.tok.encode(form, add_special_tokens=False)] if len(t) == 1})
            if not ids:
                raise ValueError(f"字母 {letter} 不是单个 token")
            self.option_ids.append(ids)
        flat = [i for ids in self.option_ids for i in ids]
        if len(set(flat)) != len(flat):
            raise ValueError("四个字母的 token 有重叠")
        self.flat_ids = set(flat)
        for letter, ids in zip(LETTERS, self.option_ids):  # 接在 "Answer:" 后面时，字母应当正好是候选里的一个 token
            tail = self.tok.encode(f"{MARKER} {letter}", add_special_tokens=False)[-1]
            if tail not in ids:
                print(f"注意：'{MARKER} {letter}' 的最后一个 token 是 {tail}，不在候选 {ids} 里")
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        self.stop_ids = set(self.eos_ids) | {self.tok.pad_token_id}
        print("模型版本：", self.revision, "；字母 token：", dict(zip(LETTERS, self.option_ids)))

    def info(self):
        t = self.torch
        return {"torch": t.__version__, "cuda": t.version.cuda, "gpu": t.cuda.get_device_name(0),
                "transformers": importlib.metadata.version("transformers"), "model_revision": self.revision}

    def chat(self, user_text):
        return self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def _encode(self, texts):
        return [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]

    def _batch(self, seqs):
        torch = self.torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, s in enumerate(seqs):  # 左侧补齐，每行最后一个位置就是答案位置
            ids[j, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[j, width - len(s):] = 1
        return ids.cuda(), mask.cuda()

    def _score_batch(self, seqs):
        torch = self.torch
        ids, mask = self._batch(seqs)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                logits_to_keep=1, use_cache=False).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("logits 出现 NaN/Inf")
        letter = torch.stack([torch.logsumexp(logits[:, i], -1) for i in self.option_ids], -1)
        log_mass = torch.logsumexp(letter, -1) - torch.logsumexp(logits, -1)
        p = torch.softmax(letter, -1)
        if not torch.isfinite(p).all() or (p.sum(-1) - 1).abs().max().item() > 1e-4:
            raise FloatingPointError("四类概率不是有限值，或和不为 1")
        top = logits.argmax(-1).tolist()
        return [{"p": p[j].tolist(), "log_mass": float(log_mass[j]), "mass": float(log_mass[j].exp()),
                 "top": self.tok.decode([top[j]]), "top_is_letter": top[j] in self.flat_ids} for j in range(len(seqs))]

    def score(self, texts):
        """每条输入（已套对话模板，以 "Answer:" 结尾）的下一个 token 上，A–D 四类的概率。显存不够时把批减半重试。"""
        seqs = self._encode(texts)
        pending = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        out = [None] * len(seqs)
        while pending:
            n = 1
            while n < len(pending) and len(seqs[pending[0]]) * (n + 1) <= self.tokens_per_batch:
                n += 1
            b = pending[:n]
            try:
                res = self._score_batch([seqs[i] for i in b])
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if n == 1:
                    raise
                self.tokens_per_batch = max(len(seqs[b[0]]), self.tokens_per_batch // 2)
                print(f"  显存不够：读概率改为每批 {self.tokens_per_batch} 个 token")
                continue
            for i, r in zip(b, res):
                out[i] = r
            pending = pending[n:]
        return out

    def _gen_batch(self, seqs, max_new):
        torch = self.torch
        cfg = self.GenerationConfig(max_new_tokens=max_new, do_sample=False, stop_strings=[MARKER],
                                    pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids)
        ids, mask = self._batch(seqs)
        with torch.no_grad():
            gen = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg, tokenizer=self.tok)
        out = []
        for j in range(len(seqs)):
            new = gen[j, ids.shape[1]:].tolist()
            n = next((k for k, t in enumerate(new) if t in self.stop_ids), len(new))
            out.append(describe_generation(self.tok.decode(new[:n], skip_special_tokens=True), n, max_new))
        return out

    def generate(self, texts, max_new):
        seqs = self._encode(texts)
        pending = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        out = [None] * len(seqs)
        while pending:
            b = pending[:self.gen_batch]
            try:
                res = self._gen_batch([seqs[i] for i in b], max_new)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if len(b) == 1:
                    raise
                self.gen_batch = max(1, len(b) // 2)
                print(f"  显存不够：生成改为每批 {self.gen_batch} 条")
                continue
            for i, r in zip(b, res):
                out[i] = r
            pending = pending[len(b):]
        return out

    def peek(self, text, n=24):
        """SMOKE 用：不替它写 "Answer:"，贪心续写几个 token，看模型实际怎么回答。"""
        cfg = self.GenerationConfig(max_new_tokens=n, do_sample=False, pad_token_id=self.tok.pad_token_id)
        ids, mask = self._batch(self._encode([text]))
        with self.torch.no_grad():
            gen = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
        return self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

    def measure(self, score_texts, gen_texts, max_new):
        """SMOKE 用：正式长度下的峰值显存和用时，并据此建议批大小。"""
        torch = self.torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        res = {"gpu_total_gb": round(total / 2 ** 30, 2), "model_gb": round(base / 2 ** 30, 2)}
        seqs = self._encode(score_texts)
        n_tok = len(seqs) * max(len(s) for s in seqs)
        res.update({"score_batch": len(seqs), "score_tokens": n_tok})
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            self._score_batch(seqs)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            per_tok = max(1, peak - base) / n_tok
            res.update({"score_peak_gb": round(peak / 2 ** 30, 2), "score_seconds": round(time.time() - t0, 2),
                        "suggest_tokens_per_batch": int(max(1024, min(32768, (0.85 * total - base) / per_tok)) // 1024 * 1024)})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res.update({"score_peak_gb": "显存不够", "score_seconds": None,
                        "suggest_tokens_per_batch": max(1024, n_tok // 2 // 1024 * 1024)})
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res.update({"gen_batch": len(gen_texts), "gen_max_new_tokens": max_new})
        try:
            gens = self._gen_batch(self._encode(gen_texts), max_new)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            per_seq = max(1, peak - base) / len(gen_texts)
            res.update({"gen_peak_gb": round(peak / 2 ** 30, 2), "gen_seconds": round(time.time() - t0, 1),
                        "gen_end": dict(Counter(g["end"] for g in gens)),
                        "gen_mean_new_tokens": round(sum(g["n_new"] for g in gens) / len(gens), 1),
                        "suggest_gen_batch": int(max(1, min(64, (0.85 * total - base) / per_seq)))})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res.update({"gen_peak_gb": "显存不够", "gen_seconds": None, "gen_end": {}, "gen_mean_new_tokens": None,
                        "suggest_gen_batch": max(1, len(gen_texts) // 2)})
        return res


def describe_generation(text, n_new, max_new):
    """生成结束的原因：写出了 "Answer:"（marker）、自己结束但没写（eos）、到了长度上限（length）。"""
    i = text.rfind(MARKER)
    if i >= 0:
        line = text[text.rfind("\n", 0, i) + 1:i]
        inline = bool(re.sub(r"(?i)final|[\s*_#>]", "", line))  # 同一行 "Answer:" 前面还有推理文字
        return {"gen": text, "n_new": n_new, "end": "marker", "marker_inline": inline}
    return {"gen": text, "n_new": n_new, "end": "length" if n_new >= max_new else "eos", "marker_inline": False}


def cot_continuation(gen):
    """保留 "Answer:" 之前的全部推理（包括同一行的），去掉 "**Final" 这类答案标记，统一接上 "\\nAnswer:"。"""
    i = gen.rfind(MARKER)
    body = gen[:i] if i >= 0 else gen
    body = re.sub(r"(?i)[\s*_#>]*(?:the\s+)?final[\s*_#>]*$", "", body).rstrip(" \t\r\n*_#>")
    return (body + "\n" if body else "") + MARKER


# ---------- 预测、评分 ----------

def sym(p_ab, p_ba):
    return [(p_ab[i] + p_ba[SWAP[i]]) / 2 for i in range(4)]


def argmax_label(p):
    return LABELS[max(range(4), key=lambda i: p[i])]


def weighted_f1(gold, pred):
    total, out = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        out += f1 * sum(g == l for g in gold) / total
    return out


def official_score(gold, pred, out_dir):
    gold, pred, out_dir = (os.path.abspath(p) for p in (gold, pred, out_dir))  # 评分脚本在官方仓库目录下运行
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", gold,
                    "--predictions", pred, "--output-dir", out_dir],
                   cwd=DICO_DIR, check=True, capture_output=True)
    s = read_json(os.path.join(out_dir, "scores.json"))
    return {k: s[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}


def mass_stats(items):
    masses = sorted(u["mass"] for u in items)
    q = lambda f: round(masses[min(len(masses) - 1, int(f * len(masses)))], 4)
    low = sorted(items, key=lambda u: u["mass"])[:5]
    return {"mean_answer_mass": round(sum(masses) / len(masses), 4), "answer_mass_p05": q(0.05),
            "answer_mass_p25": q(0.25), "answer_mass_below_0.5": sum(m < 0.5 for m in masses),
            "top_is_letter": round(sum(u["top_is_letter"] for u in items) / len(items), 4),
            "lowest_mass": [{"a": u["a"], "b": u["b"], "mass": round(u["mass"], 4), "top": u["top"]} for u in low]}


# ---------- 一个阶段 = 一种提示词 × 一个数据文件 ----------

def stage_dir(prompt, split, track):
    return os.path.join(OUT, prompt, f"{split}_track{track}")


def run_stage(backend, prompt, split, track, examples, excluded):
    d = stage_dir(prompt, split, track)
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"跳过（已完成）：{prompt} {split} track{track}")
        return
    t0 = time.time()
    rows = stage_rows(split, track, excluded)
    inputs = needed_inputs(rows)
    cache_path, gen_path = os.path.join(d, "cache.jsonl"), os.path.join(d, "gen.jsonl")
    cache = load_jsonl(cache_path)
    todo = [x for x in inputs if key_of(*x) not in cache]
    print(f"开始：{prompt} {split} track{track}：{len(rows)} 行，{len(inputs)} 个有序输入，还要算 {len(todo)} 个")
    if prompt == "cot":
        gens = load_jsonl(gen_path)
        need = [x for x in todo if key_of(*x) not in gens]
        prompts = {key_of(*x): backend.chat(build_prompt("cot", *x)) for x in need}
        need.sort(key=lambda x: -len(prompts[key_of(*x)]))
        k, shown = 0, 0
        while k < len(need):
            part = need[k:k + backend.gen_batch]
            res = backend.generate([prompts[key_of(*x)] for x in part], cot_tokens())
            items = [{"a": a, "b": b, **r} for (a, b), r in zip(part, res)]
            append_jsonl(gen_path, items)
            gens.update({key_of(x["a"], x["b"]): x for x in items})
            k += len(part)
            if k * 10 // len(need) > shown:
                shown = k * 10 // len(need)
                print(f"  已生成 {k}/{len(need)}，用时 {(time.time() - t0) / 60:.1f} 分钟")
        texts = {key_of(*x): backend.chat(build_prompt("cot", *x)) + cot_continuation(gens[key_of(*x)]["gen"])
                 for x in todo}
    else:
        texts = {key_of(*x): backend.chat(build_prompt(prompt, *x, examples)) + MARKER for x in todo}
    todo.sort(key=lambda x: -len(texts[key_of(*x)]))
    shown = 0
    for n, part in enumerate(chunks(todo, SCORE_CHUNK), 1):
        res = backend.score([texts[key_of(*x)] for x in part])
        items = [{"a": a, "b": b, **r} for (a, b), r in zip(part, res)]
        append_jsonl(cache_path, items)
        cache.update({key_of(x["a"], x["b"]): x for x in items})
        if n * SCORE_CHUNK * 10 // max(1, len(todo)) > shown:
            shown = n * SCORE_CHUNK * 10 // max(1, len(todo))
            print(f"  已读概率 {min(n * SCORE_CHUNK, len(todo))}/{len(todo)}，用时 {(time.time() - t0) / 60:.1f} 分钟")
    prob = lambda r, swap=False: cache[key_of(r["text2"], r["text1"]) if swap else key_of(r["text1"], r["text2"])]["p"]
    pred = {r["instance_id"]: {"plain": argmax_label(prob(r)), "sym": argmax_label(sym(prob(r), prob(r, True)))}
            for r in rows}
    subsets = {"all": rows} if split == "dev" else train_subsets(rows)
    ref_rows = read_rows(data_file(track, split, "reference"))
    scores = {}
    for name, sub in subsets.items():
        if not sub:
            continue
        keep = {r["instance_id"] for r in sub}
        if split == "dev":
            ref = data_file(track, split, "reference")
        else:  # 按家族成对去掉，reverse_pair_id 的互指关系不变
            ref = os.path.join(d, f"reference_{name}.csv")
            write_rows(ref, [r for r in ref_rows if r["instance_id"] in keep], list(ref_rows[0].keys()))
        scores[name] = {}
        for mode in ("plain", "sym"):
            path = os.path.join(d, f"pred_{name}_{mode}.csv")
            write_rows(path, [{"instance_id": r["instance_id"], "label": pred[r["instance_id"]][mode]} for r in sub],
                       ["instance_id", "label"])
            scores[name][mode] = ({"weighted_f1": weighted_f1([r["label"] for r in sub],
                                                              [pred[r["instance_id"]][mode] for r in sub])}
                                  if SMOKE else official_score(ref, path, os.path.join(d, f"scores_{name}_{mode}")))
    stats = {"rows": len(rows), "inputs": len(inputs), "subset_rows": {k: len(v) for k, v in subsets.items()},
             "pred_plain_counts": dict(Counter(p["plain"] for p in pred.values())),
             **mass_stats([cache[key_of(*x)] for x in inputs])}
    if prompt == "cot":
        g = [gens[key_of(*x)] for x in inputs]
        stats.update({"cot_end": dict(Counter(x["end"] for x in g)),
                      "cot_marker_inline": sum(x["marker_inline"] for x in g),
                      "cot_mean_new_tokens": round(sum(x["n_new"] for x in g) / len(g), 1),
                      "cot_max_new_tokens": cot_tokens()})
    write_json(os.path.join(d, "done.json"), {"scores": scores, "stats": stats,
                                              "minutes": round((time.time() - t0) / 60, 1)})
    print(f"完成：{prompt} {split} track{track}  {scores}  answer mass 均值 {stats['mean_answer_mass']}"
          f"  用时 {(time.time() - t0) / 60:.1f} 分钟")


def stages():
    """先跑英语训练集（用来选提示词），再跑 dev（只报告）。"""
    return [(p, "train", 1) for p in TRAIN_PROMPTS] + [(p, "dev", t) for p in PROMPTS for t in TRACKS]


def select_prompt():
    """按事先定好的规则，在训练集的 select 子集上选提示词；训练集阶段没跑完时返回 None。"""
    wf1 = {}
    for p in TRAIN_PROMPTS:
        path = os.path.join(stage_dir(p, "train", 1), "done.json")
        if not os.path.exists(path) or "select" not in read_json(path)["scores"]:
            return None
        wf1[p] = read_json(path)["scores"]["select"]["sym"]["weighted_f1"]
    best = max(TRAIN_PROMPTS, key=lambda p: (wf1[p], -TRAIN_PROMPTS.index(p)))
    sel = {"rule": SELECTION_RULE, "selected": best, "select_sym_weighted_f1": wf1}
    write_json(os.path.join(OUT, "selection.json"), sel)
    return sel


def summarize():
    results = []
    for prompt, split, track in stages():
        path = os.path.join(stage_dir(prompt, split, track), "done.json")
        if not os.path.exists(path):
            continue
        d = read_json(path)
        for subset, by_mode in d["scores"].items():
            for mode, s in by_mode.items():
                results.append({"prompt": prompt, "split": split, "track": track, "subset": subset, "mode": mode,
                                **{k: round(v, 4) for k, v in s.items()}, "answer_mass": d["stats"]["mean_answer_mass"]})
    missing = [f"{p} {s} track{t}" for p, s, t in stages()
               if not os.path.exists(os.path.join(stage_dir(p, s, t), "done.json"))]
    sel = select_prompt()
    write_json(os.path.join(OUT, "summary.json"), {"version": VERSION, "model": MODEL, "selection": sel,
                                                   "results": results, "missing": missing})
    fmt = lambda r, k: f"{r[k]:.3f}" if k in r else "  -  "
    for split, title in (("train", "英语训练集（用来选提示词）"), ("dev", "官方 dev（只报告）")):
        rs = [r for r in results if r["split"] == split]
        if rs:
            print(f"\n{title}：提示词 / 赛道 / 子集 / 预测方式：WF1、SoftCons、HardCons、answer mass")
            for r in rs:
                print(f"  {r['prompt']:<10} track{r['track']} {r['subset']:<11} {r['mode']:<5}  {fmt(r, 'weighted_f1')}"
                      f"  {fmt(r, 'soft_cons')}  {fmt(r, 'hard_cons')}  {r['answer_mass']:.3f}")
        if split == "train" and sel:
            print(f"\n按事先定好的规则选出的提示词：{sel['selected']}（select 子集对称化 WF1：{sel['select_sym_weighted_f1']}）")
    return missing


def smoke_checks(backend, examples_by_track, excluded):
    """SMOKE：看原始回答、批内一致性和正式长度的显存。格式例子取自训练集，不看 dev。"""
    checks = {}
    rows = train_subsets(stage_rows("train", 1, excluded))["select"][:3]
    for r in rows:
        text = backend.chat(build_prompt("zero", r["text1"], r["text2"]))
        s = backend.score([text + MARKER])[0]
        print(f"\n[{r['text1']} ‖ {r['text2']}] 金标 {r['label']}")
        print("  不替它写 Answer: 时的自由续写：", repr(backend.peek(text)))
        print("  写上 Answer: 后，下一个 token 最可能是：", repr(s["top"]), "；A–D 概率：",
              [round(x, 3) for x in s["p"]], "；answer mass", round(s["mass"], 3))
    long_track = 4 if 4 in TRACKS else 1
    pool = read_rows(data_file(long_track, "train", "participant_labeled"))  # 显存检查只看长度，取训练集里最长的输入
    texts = ([backend.chat(build_prompt("zero", r["text1"], r["text2"])) + MARKER for r in rows]
             + [backend.chat(build_prompt("few", r["text1"], r["text2"], examples_by_track[1])) + MARKER for r in rows])
    alone = [backend.score([t])[0]["p"] for t in texts]
    together = [x["p"] for x in backend.score(texts)]
    diff = max(abs(a - b) for pa, pb in zip(alone, together) for a, b in zip(pa, pb))
    same = sum(argmax_label(pa) == argmax_label(pb) for pa, pb in zip(alone, together))
    checks["batch_vs_alone"] = {"inputs": len(texts), "max_abs_prob_diff": round(diff, 6), "same_label": same}
    print(f"\n同一输入单独评分和放进长短混合的批里评分：四类概率最大差 {diff:.2e}，{same}/{len(texts)} 条标签相同")
    r = rows[0]
    print("\n少样本提示词（赛道 1）：\n" + build_prompt("few", r["text1"], r["text2"], examples_by_track[1]))
    g = backend.generate([backend.chat(build_prompt("cot", r["text1"], r["text2"]))], SMOKE_COT_TOKENS)[0]
    print("\ncot 生成（SMOKE 只到 64 个 token）：", repr(g["gen"]), "；结束原因", g["end"],
          "\n  接上的答案位置：", repr(cot_continuation(g["gen"])[-80:]))
    longest = sorted(pool, key=lambda x: -len(x["text1"]) - len(x["text2"]))
    few = [backend.chat(build_prompt("few", x["text1"], x["text2"], examples_by_track[long_track])) + MARKER
           for x in longest[:64]]
    width = max(len(t) for t in few) // 3 + 1  # 粗估 token 数，只用来决定这一批放几条
    few = few[:max(1, TOKENS_PER_BATCH // width)]
    cot = [backend.chat(build_prompt("cot", x["text1"], x["text2"])) for x in longest[:GEN_BATCH]]
    m = backend.measure(few, cot, COT_MAX_NEW_TOKENS)
    checks["resources"] = m
    print(f"\n正式长度检查：显存共 {m['gpu_total_gb']} GB，模型占 {m['model_gb']} GB。"
          f"\n  读概率：一批 {m['score_batch']} 条最长的少样本输入（约 {m['score_tokens']} 个 token），峰值 {m['score_peak_gb']} GB，"
          f"{m['score_seconds']} 秒；建议 TOKENS_PER_BATCH = {m['suggest_tokens_per_batch']}"
          f"\n  生成：一批 {m['gen_batch']} 条、最多 {m['gen_max_new_tokens']} 个 token，峰值 {m['gen_peak_gb']} GB，"
          f"{m['gen_seconds']} 秒，结束原因 {m['gen_end']}，平均生成 {m['gen_mean_new_tokens']} 个 token；"
          f"建议 GEN_BATCH = {m['suggest_gen_batch']}")
    write_json(os.path.join(OUT, "smoke_checks.json"), checks)


def main():
    ensure_packages()
    ensure_official_repo()
    chosen, excluded = pick_examples()
    fp = fingerprint(chosen)
    restore_previous_results(fp)
    os.makedirs(OUT, exist_ok=True)
    revision = check_fingerprint(fp)
    examples_by_track = {t: examples_for_track(t, chosen) for t in sorted(set(TRACKS) | {1})}
    write_json(os.path.join(OUT, "examples.json"), {str(t): ex for t, ex in examples_by_track.items()})
    missing = None
    try:
        backend = QwenBackend(revision)
        record_revision(backend.revision)
        record_run(backend)
        if SMOKE:
            smoke_checks(backend, examples_by_track, excluded)
        for prompt, split, track in stages():
            if split == "dev" and not os.path.exists(os.path.join(OUT, "selection.json")):
                sel = select_prompt()
                if sel:
                    print(f"\n训练集阶段完成。按事先定好的规则选出：{sel['selected']}。下面开始跑 dev（只报告）。\n")
            run_stage(backend, prompt, split, track, examples_by_track.get(track, []), excluded)
            pack()
        missing = summarize()
    finally:
        if missing is None:
            print("\n运行中途停下了，下面是已经完成的部分：")
            summarize()
        zip_path = pack()
        print("\n结果包已保存到：", zip_path)
        print("请把这个 zip 下载下来发给 Claude。还没跑完时，重新运行本 notebook 会接着跑；换平台时把 zip 放到新平台的工作目录。")


if __name__ == "__main__":
    main()
