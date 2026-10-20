# %% [markdown]
# # DiCo-NLI 第三组实验：Qwen 指令学习（零样本、少样本、CoT）
#
# 只有两个代码单元：先运行配置单元，再运行第二个单元（定义 + 推理 + 评分 + 打包），也可以直接 Run All。
#
# 对应导师清单第 1 项"大模型 instruct、Qwen"和第 2 项里的 CoT：不训练模型，只靠指令让 `Qwen/Qwen3-4B-Instruct-2507` 判断两个短语的关系。
#
# **四种提示词**（都用导师 SST-2 材料里的 Alpaca 格式，放进模型自己的对话模板；选项 A、B、C、D 依次对应 EQUIVALENCE、FORWARD、BACKWARD、NEGATIVE_OTHER）：
#
# - `zero`：零样本，只答一个字母。
# - `zero_rules`：在 `zero` 上只多一段标注口径：计数按"至少"读，读数不同则互不蕴含。这两条在英语训练集上核对过，没有例外。
# - `few`：在 `zero` 上只多 8 个训练集里的例子（每类 2 个）。例子按固定种子从英语训练集里抽，四个赛道用同一批 pair（各用本赛道的文本）。避开和 dev 文本相同的短语对，也不用盲审过的 45 个家族：它们是从模型出错的格子里抽的，其中 11 个的金标两位审阅者都不认同。
# - `cot`：在 `zero` 上改成先写推理、最后一行写 "Answer: X"，开头加导师材料里那句 "Before answering, think carefully ..."。
#
# **怎么取答案**：不解析自由文本。在模型回答的开头替它写上 "Answer:"，读下一个 token 是 A、B、C、D 的概率，归一化成四类的分布（和 IMDB 实验读答案词概率的做法相同）。`cot` 先让模型生成推理，再在推理后面接上 "Answer:"，用同样的方法读。四个字母合起来占的概率（answer mass）和"最可能的下一个 token 是不是字母"都记下来，用来检查模型是不是按要求作答。
#
# **两种预测**：`plain` 只看 (text1, text2)；`sym` 再把两句对调问一次，把对调结果的 F / B 概率互换后和原方向取平均，和编码器实验的对称化相同。相同的有序输入只算一次。
#
# **评测**：
#
# - 官方 dev，四个赛道，官方评分脚本。dev 只用来报告。
# - 英语训练集，去掉少样本例子所在的家族，只跑 `zero`、`zero_rules`、`few`。以后要在提示词之间做选择，看这里，不看 dev。
#
# **和导师 SST-2 材料的区别**：任务是四选一；答案读字母的概率，不解析回答文本，所以没有"无效回答"，改为报告 answer mass；模型在本机用 transformers 加载，不经过 Ollama 或 API；`cot` 用贪心解码，不采样。
#
# **断点续跑**：每算完一批就写进缓存，重新运行只算没算完的部分。换平台时把上一次的 `dico_llm_results_*.zip` 放到工作目录（Kaggle 上作为数据集挂到 `/kaggle/input`），会自动解压接着跑。模型版本、提示词和数据必须和原来一致，否则会停下并提示。
#
# **时间估计（T4，未实测）**：dev 四个赛道共约 6,900 个有序输入，英语训练集约 3,500 个。`zero` 约 30 分钟，`zero_rules` 约 40 分钟，`few` 约 55 分钟，`cot` 约 1.7 小时，合计约 4 小时。按 4B 模型、T4 约 10 TFLOPS（读入约 1,250 token/秒）、生成约 400 token/秒估算，每个阶段跑完会打印实际用时。按 `zero` → `zero_rules` → `few` → `cot` 的顺序跑，中途停下也能先拿到前面的结果。
#
# **SMOKE = True**：只跑赛道 1 和 4，每个文件取 24 条，`cot` 最多生成 64 个 token；几分钟内查完流程，并打印几条模型的原始输出。先跑 SMOKE，看一眼打印出来的回答再正式跑。
#
# 在 Kaggle 上要打开 Internet（克隆官方仓库、下载约 8 GB 的模型）。Colab、腾讯 Cloud Studio 直接运行；Cloud Studio 上如果下载不了 Hugging Face 的模型，先在第一个单元里加一行 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`。

# %%
import os

SMOKE = False
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROMPTS = ["zero", "zero_rules", "few", "cot"]  # 按这个顺序跑
TRACKS = [1, 2, 3, 4]
TRAIN_PROMPTS = ["zero", "zero_rules", "few"]  # 这几种还在英语训练集上跑，用来做选择
FEW_PER_LABEL = 2
EXAMPLE_SEED = 2027
COT_MAX_NEW_TOKENS = 320
TOKENS_PER_BATCH = 12288  # 读概率时一批最多多少个 token；显存不够就调小
GEN_BATCH = 32  # cot 生成时一批几条；显存不够就调小
SMOKE_ROWS = 24

WORK = os.path.abspath(os.environ.get("DICO_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."))
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
OUT = os.path.join(WORK, "dico_llm_smoke" if SMOKE else "dico_llm_results")  # SMOKE 的结果单独放，不会被当成正式结果续跑

if SMOKE:
    TRACKS, COT_MAX_NEW_TOKENS = [1, 4], 64

# %%
import csv
import glob
import hashlib
import importlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
import zipfile
from collections import Counter

VERSION = "dico_llm_instruct-1"
TRANSFORMERS_VERSION = "5.19.0"  # 和 IMDB 的 Qwen3-4B 实验相同
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LETTERS = ["A", "B", "C", "D"]
SWAP = [0, 2, 1, 3]  # 对调两句后：FORWARD ↔ BACKWARD，其余不变

HEADER = ("Below is an instruction that describes a task, paired with an input that provides "
          "further context. Write a response that appropriately completes the request.")
THINK_STEP = ("Before answering, think carefully about the question and create a step-by-step "
              "chain of thoughts to ensure a logical and accurate response.")
TASK = ("You are given two short phrases taken from image captions or news headlines. The phrases may be "
        "in English, Spanish or Basque. Decide which inference relation holds between Phrase 1 and Phrase 2.\n"
        "A. Equivalence: each phrase entails the other.\n"
        "B. Forward entailment: Phrase 1 entails Phrase 2, but Phrase 2 does not entail Phrase 1.\n"
        "C. Backward entailment: Phrase 2 entails Phrase 1, but Phrase 1 does not entail Phrase 2.\n"
        "D. Neither: neither phrase entails the other (they may be similar, related, opposite or unrelated).")
RULES = ("Labelling conventions of this dataset:\n"
         "- Counts are read as \"at least\": the phrase with the larger count entails the phrase with the "
         "smaller count (\"6 dead\" entails \"One dead\").\n"
         "- Phrases that differ only in a measured value (a magnitude, rate, percentage or date) do not "
         "entail each other.")
ANSWER_LETTER = "Answer with a single letter: A, B, C or D. No further explanation is required."
ANSWER_COT = ("Write your reasoning first, then give the final answer on the last line in the form "
              "\"Answer: X\", where X is A, B, C or D.")
MARKER = "Answer:"
# 盲审过的 45 个家族（metrics/diagnostics/blind_review/）：从模型出错的格子里抽的，其中 11 个的金标两位审阅者都不认同，不拿来当例子
REVIEWED_FAMILIES = {f"dico_phrasis_{i}" for i in (
    "0000079", "0000202", "0000267", "0000420", "0000473", "0000646", "0000670", "0000673", "0000682", "0000709",
    "0000853", "0000854", "0000919", "0000969", "0001024", "0001054", "0001158", "0001161", "0001187", "0001238",
    "0001401", "0001438", "0001566", "0001720", "0001774", "0001917", "0001942", "0001997", "0002082", "0002225",
    "0002275", "0002410", "0002420", "0002509", "0002511", "0002557", "0002578", "0002820", "0002833", "0002861",
    "0002998", "0003042", "0003079", "0003092", "0003316")}


def build_prompt(kind, a, b, examples=()):
    head = HEADER + ("\n" + THINK_STEP if kind == "cot" else "")
    instruction = "### Instruction:\n" + TASK
    if kind == "zero_rules":
        instruction += "\n" + RULES
    instruction += "\n" + (ANSWER_COT if kind == "cot" else ANSWER_LETTER)
    parts = [head, instruction]
    if kind == "few":
        parts.append("### Examples:\n" + "\n\n".join(
            f"Phrase 1: {x}\nPhrase 2: {y}\n{MARKER} {letter}" for x, y, letter in examples))
    parts.append(f"### Input:\nPhrase 1: {a}\nPhrase 2: {b}")
    parts.append("### Response:")
    return "\n\n".join(parts)


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


def restore_previous_results():
    if SMOKE or os.path.isdir(OUT):
        return
    zips = sorted(glob.glob(os.path.join(WORK, "dico_llm_results_*.zip"))
                  + glob.glob("/kaggle/input/*/dico_llm_results_*.zip"))
    if not zips:
        return
    best = max(zips, key=lambda z: sum(n.endswith(".jsonl") for n in zipfile.ZipFile(z).namelist()))
    zipfile.ZipFile(best).extractall(WORK)
    print("已从上一次的结果包恢复：", best)


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


def pick_examples():
    """少样本例子：每类 FEW_PER_LABEL 个 pair，取自英语训练集；避开和 dev 文本相同的短语对和盲审过的家族，同一家族最多用一次。"""
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
        picked = [p for p in cands if fam[p] not in used][:FEW_PER_LABEL]
        for p in picked:
            used.add(fam[p])
            chosen.append({"pair_id": p, "label": label})
    rng.shuffle(chosen)
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
        out.append((r["text1"], r["text2"], LETTERS[LABELS.index(c["label"])]))
    return out


def stage_rows(split, track, excluded_families):
    rows = read_rows(data_file(track, split, "participant_labeled"))
    if split == "train":
        fam = build_families(rows)
        rows = [r for r in rows if fam[r["pair_id"]] not in excluded_families]
    return rows[:SMOKE_ROWS] if SMOKE else rows


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
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    out[key_of(d["a"], d["b"])] = d
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


# ---------- 指纹 ----------

def fingerprint(chosen):
    files = sorted({data_file(t, s, k) for t in (1, 2, 3, 4) for s in ("train", "dev")
                    for k in ("participant_labeled", "reference")})
    return {
        "version": VERSION, "model": MODEL, "transformers": TRANSFORMERS_VERSION, "dtype": "float16",
        "prompt_text": {"header": HEADER, "think_step": THINK_STEP, "task": TASK, "rules": RULES,
                        "answer_letter": ANSWER_LETTER, "answer_cot": ANSWER_COT, "marker": MARKER},
        "examples": chosen, "few_per_label": FEW_PER_LABEL, "example_seed": EXAMPLE_SEED,
        "reviewed_families_excluded": sorted(REVIEWED_FAMILIES),
        "cot_max_new_tokens": COT_MAX_NEW_TOKENS, "decoding": "greedy",
        "smoke": SMOKE, "smoke_rows": SMOKE_ROWS if SMOKE else None,
        "data_sha256": {os.path.relpath(p, DICO_DIR): sha256(p) for p in files},
    }


def check_fingerprint(fp):
    path = os.path.join(OUT, "fingerprint.json")
    if os.path.exists(path):
        old = read_json(path)
        diff = sorted(k for k in set(fp) | set(old) if k != "model_revision" and fp.get(k) != old.get(k))
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


# ---------- 模型（唯一用到 torch 的部分） ----------

class QwenBackend:
    def __init__(self, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if not torch.cuda.is_available():
            raise SystemExit("没有检测到 GPU。请在 Kaggle / Colab 里打开 GPU（T4）再运行。")
        self.torch, self.GenerationConfig = torch, GenerationConfig
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
        self.tok.padding_side = "left"
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
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.pad_token_id}
        print("模型版本：", self.revision, "；字母 token：", dict(zip(LETTERS, self.option_ids)))

    def chat(self, user_text):
        return self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def _batch(self, seqs):
        torch = self.torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, s in enumerate(seqs):  # 左侧补齐，每行最后一个位置就是答案位置
            ids[j, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[j, width - len(s):] = 1
        return ids.cuda(), mask.cuda()

    def score(self, texts):
        """每条输入（已套对话模板）的下一个 token 上，A–D 四个字母的概率。"""
        torch = self.torch
        seqs = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        batches, cur = [], []
        for i in order:
            if cur and len(seqs[cur[0]]) * (len(cur) + 1) > TOKENS_PER_BATCH:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        out = [None] * len(seqs)
        for b in batches:
            ids, mask = self._batch([seqs[i] for i in b])
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                    logits_to_keep=1, use_cache=False).logits[:, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            if not torch.isfinite(logp).all():
                raise FloatingPointError("logits 出现 NaN/Inf")
            p = logp.exp()
            opt = torch.stack([p[:, ids_].sum(-1) for ids_ in self.option_ids], dim=-1)
            mass = opt.sum(-1)
            top = p.argmax(-1).tolist()
            for j, i in enumerate(b):
                out[i] = {"p": (opt[j] / mass[j]).tolist(), "mass": float(mass[j]),
                          "top": self.tok.decode([top[j]]), "top_is_letter": top[j] in self.flat_ids}
        return out

    def generate(self, texts, max_new):
        torch = self.torch
        seqs = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        cfg = self.GenerationConfig(max_new_tokens=max_new, do_sample=False, stop_strings=[MARKER],
                                    pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.model.generation_config.eos_token_id)
        out = [None] * len(seqs)
        for b in chunks(order, GEN_BATCH):
            ids, mask = self._batch([seqs[i] for i in b])
            with torch.no_grad():
                gen = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg,
                                          tokenizer=self.tok)
            for j, i in enumerate(b):
                new = gen[j, ids.shape[1]:].tolist()
                n = next((k for k, t in enumerate(new) if t in self.stop_ids), len(new))
                text = self.tok.decode(new[:n], skip_special_tokens=True)
                out[i] = {"gen": text, "n_new": n, "truncated": n >= max_new and MARKER not in text}
        return out

    def peek(self, text, n=24):
        """SMOKE 用：贪心续写几个 token，看模型实际怎么回答。"""
        cfg = self.GenerationConfig(max_new_tokens=n, do_sample=False, pad_token_id=self.tok.pad_token_id)
        ids, mask = self._batch([self.tok(text, add_special_tokens=False)["input_ids"]])
        with self.torch.no_grad():
            gen = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
        return self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)


def cot_continuation(gen):
    """推理接上 "Answer:"：去掉模型自己写的最后一个答案行，统一换成 "\\nAnswer:"，再读下一个 token。"""
    i = gen.rfind(MARKER)
    body = gen[:gen.rfind("\n", 0, i) + 1] if i >= 0 else gen
    body = body.rstrip()
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


def reference_for(split, track, rows, stage_dir):
    """dev 用官方 reference；训练集去掉了例子所在的家族，写一份同样去掉这些行的 reference（成对去掉，互指关系不变）。"""
    ref = data_file(track, split, "reference")
    if split == "dev":
        return ref
    keep = {r["instance_id"] for r in rows}
    ref_rows = read_rows(ref)
    sub = [r for r in ref_rows if r["instance_id"] in keep]
    path = os.path.join(stage_dir, "reference_subset.csv")
    write_rows(path, sub, list(ref_rows[0].keys()))
    return path


# ---------- 一个阶段 = 一种提示词 × 一个数据文件 ----------

def run_stage(backend, prompt, split, track, examples, excluded):
    d = os.path.join(OUT, prompt, f"{split}_track{track}")
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"跳过（已完成）：{prompt} {split} track{track}")
        return
    t0 = time.time()
    rows = stage_rows(split, track, excluded)
    inputs = needed_inputs(rows)
    cache = load_jsonl(os.path.join(d, "cache.jsonl"))
    todo = [x for x in inputs if key_of(*x) not in cache]
    print(f"开始：{prompt} {split} track{track}：{len(rows)} 行，{len(inputs)} 个有序输入，还要算 {len(todo)} 个")
    if prompt == "cot":
        gens = load_jsonl(os.path.join(d, "gen.jsonl"))
        need = [x for x in todo if key_of(*x) not in gens]
        for part in chunks(need, 256):
            res = backend.generate([backend.chat(build_prompt("cot", a, b)) for a, b in part], COT_MAX_NEW_TOKENS)
            items = [{"a": a, "b": b, **r} for (a, b), r in zip(part, res)]
            append_jsonl(os.path.join(d, "gen.jsonl"), items)
            gens.update({key_of(x["a"], x["b"]): x for x in items})
            print(f"  已生成 {len(gens)}/{len(inputs)}，用时 {(time.time() - t0) / 60:.1f} 分钟")
        texts = {key_of(a, b): backend.chat(build_prompt("cot", a, b)) + cot_continuation(gens[key_of(a, b)]["gen"])
                 for a, b in todo}
    else:
        texts = {key_of(a, b): backend.chat(build_prompt(prompt, a, b, examples)) + MARKER for a, b in todo}
    for part in chunks(todo, 512):
        res = backend.score([texts[key_of(a, b)] for a, b in part])
        items = [{"a": a, "b": b, **r} for (a, b), r in zip(part, res)]
        append_jsonl(os.path.join(d, "cache.jsonl"), items)
        cache.update({key_of(x["a"], x["b"]): x for x in items})
        print(f"  已读概率 {sum(key_of(*x) in cache for x in inputs)}/{len(inputs)}，用时 {(time.time() - t0) / 60:.1f} 分钟")
    ids = [r["instance_id"] for r in rows]
    gold = [r["label"] for r in rows]
    preds = {
        "plain": [argmax_label(cache[key_of(r["text1"], r["text2"])]["p"]) for r in rows],
        "sym": [argmax_label(sym(cache[key_of(r["text1"], r["text2"])]["p"], cache[key_of(r["text2"], r["text1"])]["p"]))
                for r in rows],
    }
    ref = None if SMOKE else reference_for(split, track, rows, d)
    scores = {}
    for mode, labels in preds.items():
        path = os.path.join(d, f"pred_{mode}.csv")
        write_rows(path, [{"instance_id": i, "label": l} for i, l in zip(ids, labels)], ["instance_id", "label"])
        scores[mode] = ({"weighted_f1": weighted_f1(gold, labels)} if SMOKE
                        else official_score(ref, path, os.path.join(d, f"scores_{mode}")))
    used = [cache[key_of(*x)] for x in inputs]
    stats = {
        "rows": len(rows), "inputs": len(inputs),
        "mean_answer_mass": round(sum(u["mass"] for u in used) / len(used), 4),
        "top_is_letter": round(sum(u["top_is_letter"] for u in used) / len(used), 4),
        "pred_plain_counts": dict(Counter(preds["plain"])),
    }
    if prompt == "cot":
        g = [gens[key_of(*x)] for x in inputs]
        stats.update({"cot_truncated": sum(x["truncated"] for x in g),
                      "cot_wrote_marker": sum(MARKER in x["gen"] for x in g),
                      "cot_mean_new_tokens": round(sum(x["n_new"] for x in g) / len(g), 1)})
    write_json(os.path.join(d, "done.json"), {"scores": scores, "stats": stats,
                                              "minutes": round((time.time() - t0) / 60, 1)})
    print(f"完成：{prompt} {split} track{track}  {scores}  answer mass {stats['mean_answer_mass']}"
          f"  用时 {(time.time() - t0) / 60:.1f} 分钟")


def stages():
    out = []
    for prompt in PROMPTS:
        out += [(prompt, "dev", t) for t in TRACKS]
        if prompt in TRAIN_PROMPTS:
            out.append((prompt, "train", 1))
    return out


def summarize():
    rows = []
    for prompt, split, track in stages():
        path = os.path.join(OUT, prompt, f"{split}_track{track}", "done.json")
        if not os.path.exists(path):
            continue
        d = read_json(path)
        for mode, s in d["scores"].items():
            rows.append({"prompt": prompt, "split": split, "track": track, "mode": mode,
                         **{k: round(v, 4) for k, v in s.items()}, "answer_mass": d["stats"]["mean_answer_mass"]})
    missing = [f"{p} {s} track{t}" for p, s, t in stages()
               if not os.path.exists(os.path.join(OUT, p, f"{s}_track{t}", "done.json"))]
    write_json(os.path.join(OUT, "summary.json"), {"version": VERSION, "model": MODEL, "results": rows,
                                                   "missing": missing})
    print("\n提示词 / 数据 / 赛道 / 预测方式：WF1、SoftCons、HardCons、answer mass")
    for r in rows:
        print(f"  {r['prompt']:<10} {r['split']:<5} track{r['track']} {r['mode']:<5}  "
              f"{r.get('weighted_f1', float('nan')):.3f}  {r.get('soft_cons', float('nan')):.3f}  "
              f"{r.get('hard_cons', float('nan')):.3f}  {r['answer_mass']:.3f}")
    return missing


def show_samples(backend, examples_by_track):
    """SMOKE：打印几条原始回答，确认模型按要求只答一个字母。"""
    rows = stage_rows("dev", 1, set())[:3]
    for r in rows:
        text = backend.chat(build_prompt("zero", r["text1"], r["text2"]))
        s = backend.score([text + MARKER])[0]
        print(f"\n[{r['text1']} ‖ {r['text2']}] 金标 {r['label']}")
        print("  不替它写 Answer: 时的自由续写：", repr(backend.peek(text)))
        print("  写上 Answer: 后，下一个 token 最可能是：", repr(s["top"]), "；A–D 概率：",
              [round(x, 3) for x in s["p"]], "；answer mass", round(s["mass"], 3))
    r = rows[0]
    print("\n少样本提示词（赛道 1）：\n" + build_prompt("few", r["text1"], r["text2"], examples_by_track[1]))
    g = backend.generate([backend.chat(build_prompt("cot", r["text1"], r["text2"]))], COT_MAX_NEW_TOKENS)[0]
    print("\ncot 生成：", repr(g["gen"]), "\n  接上的答案位置：", repr(cot_continuation(g["gen"])[-60:]))


def pack():
    stamp = time.strftime("%Y%m%d-%H%M")
    zip_path = os.path.join(WORK, f"{os.path.basename(OUT)}_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
            if os.path.isfile(path) and not path.endswith(".tmp"):
                z.write(path, os.path.relpath(path, WORK))
    return zip_path


def main():
    ensure_packages()
    ensure_official_repo()
    restore_previous_results()
    os.makedirs(OUT, exist_ok=True)
    chosen, excluded = pick_examples()
    revision = check_fingerprint(fingerprint(chosen))
    examples_by_track = {t: examples_for_track(t, chosen) for t in sorted(set(TRACKS) | {1})}
    write_json(os.path.join(OUT, "examples.json"), {str(t): ex for t, ex in examples_by_track.items()})
    backend = QwenBackend(revision)
    record_revision(backend.revision)
    if SMOKE:
        show_samples(backend, examples_by_track)
    for prompt, split, track in stages():
        run_stage(backend, prompt, split, track, examples_by_track.get(track, []), excluded)
    missing = summarize()
    zip_path = pack()
    print("\n结果包已保存到：", zip_path)
    if missing:
        print("还有没跑完的部分：", missing, "。重新运行本 notebook 会接着跑；换平台时把这个 zip 放到新平台的工作目录。")
    print("请把这个 zip 下载下来发给 Claude。")


if __name__ == "__main__":
    main()
