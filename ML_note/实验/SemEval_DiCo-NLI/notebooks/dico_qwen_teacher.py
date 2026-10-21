# %% [markdown]
# # DiCo-NLI：导出 Qwen 老师的四类概率（蒸馏第 1 步）
#
# 只有两个代码单元，直接 Run All。**不训练**，只读概率，约 30 分钟（大部分是下载模型和每个种子加载一次）。
#
# **做什么**：载入第四组英语 3 轮 SFT 的三个模型（seed 42 / 43 / 44，各自选中的那一轮的 LoRA 权重），给 fold 0 划分里的 2,190 行训练部分的每个有序输入（两个方向都打）读出 A–D 四类概率。这些概率之后当作软标签，蒸馏给 mDeBERTa。第 s 个种子的学生用第 s 个种子的老师。
#
# 老师训练时见过这 2,190 行，所以概率会偏尖。这是事先定好的第 1 步：最便宜，先看有没有用；没用再换成"交叉拟合"的老师。
#
# **要先挂上第四组那次运行的输出**：在 Kaggle 右边点 **Add Input → Notebook Output Files**，选 SFT notebook 里跑英语 3 轮的那个版本（`dico_qwen_sft-3`，`MAX_EPOCHS = 3`）。本 notebook 会在 `/kaggle/input` 下自己找 `dico_qwen_sft_results`，只认 fingerprint 和已核验的那份逐字节一致（划分、训练设置、LoRA、提示词、数据哈希都在里面）、运行代码相同、三个种子的权重都在的那一份；还会核对当前官方数据的哈希和 fold 0 的行。找不到会说明原因。每个种子的 LoRA 权重的 SHA256 记进结果指纹，续跑时权重变了会停下。GPU 选 T4，打开 Internet（下载底座模型）。
#
# **导出的是什么**：训练部分 2,190 行去重后、两个方向一共 2,502 个有序输入的四类概率；学生训练时按原始的 2,190 行映射回去（保留重复），多出来的反向输入不用。内层验证的概率另存一份，只作诊断。
#
# **先核对，对不上就停下**：每个种子载入权重后，
#
# 1. 模型里的 LoRA 参数和文件逐位相同；
# 2. 重新给内层验证（244 行）读概率，和训练时保存的逐条概率（`val_probs/epoch<轮>.json`）比较；
# 3. 再给 fold 0 / dev 缓存里的前 128 个输入读概率，和那次运行保存的比较。
#
# 第 2、3 两项逐条预测一致率都要不低于 98%，否则停下。这两项也会证明这个环境和训练时算得一样。
#
# **SMOKE = True**：只跑 seed 42，核对照常，导出只做前 64 个输入，十分钟左右。这个 notebook 不训练，核对没过就会自己停下，所以也可以直接正式运行。
#
# 结果包是 `dico_qwen_teacher_results_<本次开始时间>.zip`，里面是每个种子的 `teacher.jsonl`（每行一个有序输入的四类概率）和核对记录，不含模型权重。

# %%
import os
import sys
from pathlib import Path

if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
    raise RuntimeError("CUDA 已初始化。请重启会话，再从头运行。")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SMOKE = False
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEEDS = [42, 43, 44]
HOLDOUT_FOLD = 0
EVAL_TOKENS_PER_BATCH = 8192
SOURCE_VERSION = "dico_qwen_sft-3"   # 第四组英语 3 轮那次运行
SOURCE_MAX_EPOCHS = 3
SOURCE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SOURCE_CODE_SHA256 = "2863de05aa757a80170da508d950ba8fa2dc2be579ef918642b249eb45f899ad"
# 第四组正式结果的 fingerprint.json（已复算核验，仓库 metrics/qwen_sft/）按键排序后 JSON 的 SHA256：划分、训练设置、LoRA、提示词、数据哈希都在里面
SOURCE_FINGERPRINT_SHA256 = "fcb2485b743956822b75c3703943ec806bc22ba955897c87082d0e6b8608566c"

ON_KAGGLE = Path("/kaggle/working").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
WORK = os.path.abspath(os.environ.get("DICO_WORK", "/kaggle/working" if ON_KAGGLE else "."))
os.makedirs(WORK, exist_ok=True)
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
SOURCE_ROOTS = [r for r in [os.environ.get("DICO_SFT_SOURCE"), "/kaggle/input", WORK] if r]
OUT = os.path.join(WORK, "dico_qwen_teacher_smoke" if SMOKE else "dico_qwen_teacher_results")
if SMOKE:
    SEEDS = SEEDS[:1]
print("运行平台：", "Kaggle" if ON_KAGGLE else "其他 Linux", "；结果目录：", WORK)

# %%
import csv
import gc
import glob
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import time
import zipfile
from collections import Counter

VERSION = "dico_qwen_teacher-3"
PINS = {"transformers": "5.19.0", "peft": "0.21.2", "accelerate": "1.15.0", "datasets": "5.1.0"}  # 和 IMDB 相同
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LETTERS = ["A", "B", "C", "D"]
SWAP = [0, 2, 1, 3]  # 对调两句后：FORWARD ↔ BACKWARD，其余不变
METRICS = ["weighted_f1", "soft_cons", "hard_cons"]
REFERENCE_FIELDS = ["instance_id", "pair_id", "reverse_pair_id", "label"]  # 官方评分脚本读金标时要的列
FOLDS, FOLD_SEED, INNER_FRACTION = 5, 2027, 0.1  # 和第二组 notebook 相同
RUN_STAMP = time.strftime("%Y%m%d-%H%M")

# 和第三组 notebook（dico_llm_instruct，VERSION 2）的 zero 提示词逐字相同
HEADER = ("Below is an instruction that describes a task, paired with an input that provides "
          "further context. Write a response that appropriately completes the request.")
TASK = ("You are given two short phrases taken from image captions or news headlines. The phrases may be "
        "in English, Spanish or Basque. Decide which inference relation holds between Phrase 1 and Phrase 2.\n"
        "A. Equivalence: each phrase entails the other.\n"
        "B. Forward entailment: Phrase 1 entails Phrase 2, but Phrase 2 does not entail Phrase 1.\n"
        "C. Backward entailment: Phrase 2 entails Phrase 1, but Phrase 1 does not entail Phrase 2.\n"
        "D. Other: none of A, B or C applies.")
ANSWER_LETTER = "Answer with a single letter: A, B, C or D. No further explanation is required."
MARKER = "Answer:"

SMOKE_INPUTS = 64
CHECK_CACHE_INPUTS = 128
MIN_AGREEMENT = 0.98

# ---------- 以下函数和类从 dico_qwen_sft.py（dico_qwen_sft-3）原样拷贝，保证提示词、划分和读概率完全一致 ----------

def build_prompt(a, b):
    return "\n\n".join([HEADER, "### Instruction:\n" + TASK + "\n" + ANSWER_LETTER,
                        f"### Input:\nPhrase 1: {a}\nPhrase 2: {b}", "### Response:"])


def site_package_dirs():
    import site
    import sysconfig
    dirs = [sysconfig.get_paths()[k] for k in ("purelib", "platlib")] + list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    return [Path(d).resolve() for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def remove_torchao():
    """先用 pip 卸载 torchao；卸载后 site-packages 里如果还残留它的包目录或 dist-info（PEFT 读到会报错，IMDB 实验遇到过），
    只删这两种，而且只在 site-packages 目录里找。"""
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    for root in site_package_dirs():
        for path in [root / "torchao", *root.glob("torchao-*.dist-info")]:
            path = path.resolve()
            if path.parent == root and path.is_dir():
                print("  删除 torchao 的残留：", path)
                shutil.rmtree(path)


def ensure_packages():
    have = {}
    for name in PINS:
        try:
            have[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have[name] = None
    if have == PINS:
        return
    loaded = [n for n in PINS if n in sys.modules]
    if loaded:
        raise SystemExit(f"这个内核里已经加载了 {loaded}，版本不对。请重启会话，再从头运行。")
    remove_torchao()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[f"{k}=={v}" for k, v in PINS.items()],
                    "protobuf>=4.25,<8"], check=True)
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
        k = tuple(sorted((norm(r["text1"]), norm(r["text2"]))))
        if k in first:
            a, b = find(p), find(first[k])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[k] = p
    return {p: find(p) for p in parent}


def assign_folds(rows, fam_of):
    """家族随机排序后依次放进当前行数最少的折，和第二组 notebook 相同。"""
    size = Counter(fam_of[r["pair_id"]] for r in rows)
    fams = sorted(size)
    random.Random(FOLD_SEED).shuffle(fams)
    load, fold_of = [0] * FOLDS, {}
    for f in fams:
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        fold_of[f] = k
        load[k] += size[f]
    return fold_of


def pick_families(families, fraction, seed):
    fams = sorted(set(families))
    random.Random(seed).shuffle(fams)
    return set(fams[: max(1, int(len(fams) * fraction))])


def make_split():
    """家族和折都按英语训练集算（四个赛道的 pair_id 相同）；内层验证的种子和第二组 fold 0 的折外训练相同。"""
    train1 = read_rows(data_file(1, "train", "participant_labeled"))
    fam_of = build_families(train1)
    fold_of = assign_folds(train1, fam_of)
    rest = {fam_of[r["pair_id"]] for r in train1 if fold_of[fam_of[r["pair_id"]]] != HOLDOUT_FOLD}
    inner = pick_families(rest, INNER_FRACTION, FOLD_SEED * 100 + HOLDOUT_FOLD + 1)
    return fam_of, fold_of, inner


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
                print(f"  {os.path.basename(path)} 最后一行不完整（上次被打断），已备份到 {backup} 并截掉")
                return out
            out[key_of(d["a"], d["b"])] = d
            good_end = end
        pos = end
    if data and not data.endswith(b"\n"):
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



class QwenSFT:
    def __init__(self, revision=None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():
            raise SystemExit("没有检测到 GPU。请在 Kaggle / Colab 里打开 GPU（T4）再运行。")
        if torch.cuda.device_count() != 1:
            raise SystemExit("应只看到一张 GPU。请重启会话，确认第一个代码单元最先运行。")
        self.torch = torch
        try:
            bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            bf16 = torch.cuda.is_bf16_supported()
        self.bf16, self.dtype = bf16, (torch.bfloat16 if bf16 else torch.float16)  # T4 不支持 bf16，用 fp16
        self.path = snapshot_download(MODEL, revision=revision)  # 先固定快照，再从这个本地快照加载
        self.revision = os.path.basename(os.path.normpath(self.path))
        if revision and self.revision != revision:
            raise SystemExit(f"下载到的快照 {self.revision} 和上次的 {revision} 不同")
        self.tok = AutoTokenizer.from_pretrained(self.path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.option_ids = []
        for letter in LETTERS:  # 读概率时 "A" 和 " A" 两种写法都算 A，和第三组相同
            ids = sorted({t[0] for form in (letter, " " + letter)
                          for t in [self.tok.encode(form, add_special_tokens=False)] if len(t) == 1})
            self.option_ids.append(ids)
        self.flat_ids = {i for ids in self.option_ids for i in ids}
        assert len(self.flat_ids) == sum(len(x) for x in self.option_ids), "四个字母的 token 有重叠"
        self.target_ids = []  # 训练目标：接在 "Answer:" 后面的 " A" 等
        probe = self.chat(build_prompt("a dog", "an animal")) + MARKER
        probe_ids = self.tok(probe, add_special_tokens=False)["input_ids"]
        for letter in LETTERS:
            t = self.tok.encode(" " + letter, add_special_tokens=False)
            joint = self.tok(probe + " " + letter, add_special_tokens=False)["input_ids"]
            if len(t) != 1 or joint != probe_ids + t:
                raise ValueError(f"答案 ' {letter}' 和提示词结尾分词时会合并，训练位置会错位")
            self.target_ids.append(t[0])
        print("模型快照：", self.revision, "；读概率的字母 token：", dict(zip(LETTERS, self.option_ids)),
              "；训练目标 token：", dict(zip(LETTERS, self.target_ids)))
        self.model = None

    def info(self):
        t = self.torch
        return {"torch": t.__version__, "cuda": t.version.cuda, "gpu": t.cuda.get_device_name(0),
                "model_revision": self.revision, "dtype": str(self.dtype),
                **{k: importlib.metadata.version(k) for k in PINS}}

    def chat(self, user_text):
        return self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def encode(self, a, b):
        return self.tok(self.chat(build_prompt(a, b)) + MARKER, add_special_tokens=False)["input_ids"]

    def memory(self):
        c = self.torch.cuda
        return {"allocated_gib": round(c.memory_allocated() / 2 ** 30, 2), "reserved_gib": round(c.memory_reserved() / 2 ** 30, 2)}

    def load_base(self):
        """每次都重新加载底座。显存记在 load_memory 里：加载前应接近 0，否则说明上一个模型没释放。"""
        from transformers import AutoModelForCausalLM
        self.release()
        before = self.memory()
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=self.dtype, device_map={"": 0},
                                                          attn_implementation="sdpa")
        self.model.config.use_cache = False
        self.load_memory = {"before_load": before, "after_load": self.memory()}
        print(f"    加载模型：加载前已分配显存 {before['allocated_gib']} GiB，加载后 {self.load_memory['after_load']['allocated_gib']} GiB",
              flush=True)
        return self.model

    def release(self):
        self.model = None
        gc.collect()  # Trainer 和回调之间有循环引用，要 gc 才能释放显存
        self.torch.cuda.empty_cache()

    def _last_logits(self, model, ids, mask):
        """右侧补齐，每行最后一个真实 token 的位置 = 有效长度 − 1；只为这些位置算 LM 头（和 IMDB 相同）。"""
        last = mask.sum(dim=1) - 1
        keep = self.torch.unique(last)
        out = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep, use_cache=False)
        col = self.torch.searchsorted(keep, last)
        return out.logits[self.torch.arange(ids.size(0), device=ids.device), col]

    def _pad(self, seqs):
        torch = self.torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[j, :len(s)] = 1
        return ids, mask

    def score(self, pairs):
        """每个有序输入在 "Answer:" 之后，A–D 四类的概率（每个字母的两种写法先 logsumexp，再在四类上 softmax）。"""
        torch = self.torch
        model = self.model
        was_training = model.training
        model.eval()
        seqs = [self.encode(a, b) for a, b in pairs]
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        out = [None] * len(seqs)
        i = 0
        with torch.no_grad():
            while i < len(order):
                n = 1
                while i + n < len(order) and len(seqs[order[i]]) * (n + 1) <= EVAL_TOKENS_PER_BATCH:
                    n += 1
                b = order[i:i + n]
                ids, mask = self._pad([seqs[k] for k in b])
                with torch.autocast("cuda", dtype=self.dtype):
                    logits = self._last_logits(model, ids.cuda(), mask.cuda()).float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("logits 出现 NaN/Inf")
                letter = torch.stack([torch.logsumexp(logits[:, x], -1) for x in self.option_ids], -1)
                log_mass = torch.logsumexp(letter, -1) - torch.logsumexp(logits, -1)
                p = torch.softmax(letter, -1)
                top = logits.argmax(-1).tolist()
                for j, k in enumerate(b):
                    out[k] = {"p": p[j].tolist(), "mass": float(log_mass[j].exp()), "top_is_letter": top[j] in self.flat_ids}
                i += n
        model.train(was_training)
        return out


# ---------- 以上是拷贝的部分 ----------


def track1_parts(split):
    """和 SFT 的 track_parts(1, split, smoke=False) 相同：fit / val / test。"""
    fam_of, fold_of, inner = split
    parts = {"fit": [], "val": [], "test": []}
    for r in read_rows(data_file(1, "train", "participant_labeled")):
        f = fam_of[r["pair_id"]]
        parts["test" if fold_of[f] == HOLDOUT_FOLD else "val" if f in inner else "fit"].append(r)
    return parts


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_candidate(d):
    """返回 (不符合的原因列表, 每个种子的 LoRA 权重 SHA256)。"""
    fp = read_json(os.path.join(d, "fingerprint.json"))
    why, shas = [], {}
    if canonical_sha256(fp) != SOURCE_FINGERPRINT_SHA256:
        why.append(f"fingerprint 和已核验的那份不同（版本 {fp.get('version')}，轮数 {fp.get('train', {}).get('max_epochs')}，"
                   f"快照 {str(fp.get('model_revision'))[:8]}）")
        return why, shas
    codes = {r.get("code_sha256") for r in read_json(os.path.join(d, "run_meta.json"))}
    if codes != {SOURCE_CODE_SHA256}:
        why.append(f"运行代码的哈希是 {sorted(map(str, codes))}")
    for s in SEEDS:
        sd = os.path.join(d, "track1", f"seed{s}")
        if not os.path.exists(os.path.join(sd, "done.json")):
            why.append(f"没有 seed{s} 的 done.json")
            continue
        ep = read_json(os.path.join(sd, "done.json"))["best_epoch"]
        f = os.path.join(sd, "adapters", f"epoch{ep}", "adapter_model.safetensors")
        if not os.path.exists(f):
            why.append(f"没有 seed{s} 第 {ep} 轮的 LoRA 权重")
            continue
        shas[s] = sha256(f)
    return why, shas


def find_source():
    """在挂上的输入里找第四组英语 3 轮那次运行的结果目录：fingerprint 要和已核验的那份逐字节对上，运行代码相同，三个种子的权重都在。
    找到不止一份时，它们的权重必须完全相同，否则停下。"""
    seen, problems, good = [], [], []
    for root in SOURCE_ROOTS:
        for fp_path in sorted(glob.glob(os.path.join(root, "**", "dico_qwen_sft_results", "fingerprint.json"), recursive=True)):
            d = os.path.realpath(os.path.dirname(fp_path))
            if d in seen:
                continue
            seen.append(d)
            why, shas = check_candidate(d)
            if why:
                problems.append(f"{d}：{'；'.join(why)}")
            else:
                good.append((d, shas))
    if not good:
        raise SystemExit("没找到第四组英语 3 轮那次运行的结果（含 LoRA 权重）。请在右边 Add Input → Notebook Output Files 里"
                         "挂上那个版本。找到但不符合的：\n  " + "\n  ".join(problems or ["（一个都没找到）"]))
    if len({json.dumps(s, sort_keys=True) for _, s in good}) > 1:
        raise SystemExit("找到不止一份符合的结果，但它们的 LoRA 权重不同，不知道该用哪份：\n  " + "\n  ".join(d for d, _ in good))
    d, shas = good[0]
    print("找到第四组的结果：", d)
    return d, read_json(os.path.join(d, "fingerprint.json")), shas


def check_data_and_split(src, src_fp, parts):
    """当前官方数据的哈希要和来源记录的相同；fold 0 的行和来源保存的金标子集逐行相同；训练、验证行数和来源记录的相同。"""
    for rel, h in src_fp["data_sha256"].items():
        if sha256(os.path.join(DICO_DIR, rel)) != h:
            raise SystemExit(f"官方数据 {rel} 和第四组运行时不同")
    for s in SEEDS:
        sd = os.path.join(src, "track1", f"seed{s}")
        ids = [r["instance_id"] for r in read_rows(os.path.join(sd, "reference_fold0.csv"))]
        if sorted(ids) != sorted(r["instance_id"] for r in parts["test"]):
            raise SystemExit(f"seed{s}：fold 0 的行和第四组的不同")
        t = read_json(os.path.join(sd, "trained.json"))
        if (t["fit_rows"], t["val_rows"]) != (len(parts["fit"]), len(parts["val"])):
            raise SystemExit(f"seed{s}：训练 / 验证行数和第四组记录的不同")
        if len(read_json(os.path.join(sd, "val_probs", f"epoch{t['best_epoch']}.json"))) != len(dict.fromkeys((r["text1"], r["text2"]) for r in parts["val"])):
            raise SystemExit(f"seed{s}：保存的内层验证概率条数不对")


def load_teacher(backend, fp, adapter_file):
    """按训练时同样的方式建 LoRA 模型、载入权重，并核对每个张量和文件逐位相同（和 SFT 里 load_epoch 的做法相同）。"""
    torch = backend.torch
    from peft import LoraConfig, TaskType, cast_mixed_precision_params, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    lora = fp["lora"]
    base = backend.load_base()
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora["r"], lora_alpha=lora["alpha"],
                                            lora_dropout=lora["dropout"], bias="none", target_modules=lora["targets"]))
    cast_mixed_precision_params(model, dtype=backend.dtype)
    backend.model = model
    saved = load_file(adapter_file)
    expected = get_peft_model_state_dict(model)
    missing, extra = sorted(set(expected) - set(saved)), sorted(set(saved) - set(expected))
    shapes = [k for k in saved if k in expected and tuple(saved[k].shape) != tuple(expected[k].shape)]
    if missing or extra or shapes:
        raise SystemExit(f"LoRA 文件和模型对不上：缺 {missing[:3]}，多 {extra[:3]}，形状不同 {shapes[:3]}")
    res = set_peft_model_state_dict(model, saved)
    if getattr(res, "unexpected_keys", None):
        raise SystemExit(f"载入 LoRA 权重时有对不上的 key：{res.unexpected_keys[:5]}")
    now = get_peft_model_state_dict(model)
    differ = [k for k in saved if not torch.equal(now[k].detach().cpu().to(saved[k].dtype), saved[k])]
    if differ:
        raise SystemExit(f"载入后有 {len(differ)} 个张量和文件不同，例如 {differ[:3]}")
    model.eval()
    return len(saved)


def compare(new, old):
    top = lambda p: max(range(4), key=lambda i: p[i])
    agree = sum(top(p) == top(q) for p, q in zip(new, old)) / len(old)
    diff = max(abs(a - b) for p, q in zip(new, old) for a, b in zip(p, q))
    return round(agree, 4), round(diff, 6)


def valid_prob(p):
    return isinstance(p, list) and len(p) == 4 and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in p) and abs(sum(p) - 1) < 1e-3


def export_seed(backend, src, fp, seed, parts, adapter_sha):
    d = os.path.join(OUT, "track1", f"seed{seed}")
    if os.path.exists(os.path.join(d, "done.json")):
        if read_json(os.path.join(d, "done.json")).get("adapter_sha256") != adapter_sha:
            raise SystemExit(f"seed{seed} 已完成的导出用的是另一份 LoRA 权重。请删除 {d} 后重新运行。")
        print(f"跳过（已完成）：seed{seed}")
        return
    t0 = time.time()
    s_dir = os.path.join(src, "track1", f"seed{seed}")
    best = read_json(os.path.join(s_dir, "done.json"))["best_epoch"]
    adapter = os.path.join(s_dir, "adapters", f"epoch{best}", "adapter_model.safetensors")
    if sha256(adapter) != adapter_sha:
        raise SystemExit(f"seed{seed} 的 LoRA 权重在运行中变了")
    os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "adapter.json")  # 这个目录里的缓存只属于这一份权重
    if os.path.exists(meta) and read_json(meta)["adapter_sha256"] != adapter_sha:
        raise SystemExit(f"{d} 里的缓存是另一份 LoRA 权重导出的。请删除这个目录后重新运行。")
    write_json(meta, {"adapter_sha256": adapter_sha, "best_epoch": best})
    print(f"开始：seed{seed}（第 {best} 轮的 LoRA 权重，SHA256 {adapter_sha[:12]}）", flush=True)
    n_tensors = load_teacher(backend, fp, adapter)
    # 核对 1：内层验证，顺序和训练时相同
    val_inputs = list(dict.fromkeys((r["text1"], r["text2"]) for r in parts["val"]))
    val_new = [x["p"] for x in backend.score(val_inputs)]
    val_agree, val_diff = compare(val_new, read_json(os.path.join(s_dir, "val_probs", f"epoch{best}.json")))
    # 核对 2：那次运行 fold 0 / dev 缓存的前若干个输入
    cached = [json.loads(line) for line in open(os.path.join(s_dir, "cache.jsonl"), encoding="utf-8")][:CHECK_CACHE_INPUTS]
    cache_new = [x["p"] for x in backend.score([(c["a"], c["b"]) for c in cached])]
    cache_agree, cache_diff = compare(cache_new, [c["p"] for c in cached])
    checks = {"tensors_equal": n_tensors, "val_argmax_agreement": val_agree, "val_max_prob_diff": val_diff,
              "cache_argmax_agreement": cache_agree, "cache_max_prob_diff": cache_diff}
    print(f"    核对：{n_tensors} 个张量和文件逐位相同；内层验证逐条一致 {val_agree:.1%}（概率最大差 {val_diff:.2g}）；"
          f"fold 0 / dev 缓存前 {len(cached)} 个一致 {cache_agree:.1%}（最大差 {cache_diff:.2g}）", flush=True)
    if val_agree < MIN_AGREEMENT or cache_agree < MIN_AGREEMENT:
        raise SystemExit("读出的概率和训练时差得太多，先停下。请把输出发给 Claude。")
    # 导出：内层验证（只作诊断，学生训练不用）和训练部分，分别补齐缺的
    cache_path = os.path.join(d, "teacher.jsonl")
    have = load_jsonl(cache_path)
    val_missing = [(x, p) for x, p in zip(val_inputs, val_new) if key_of(*x) not in have]
    append_jsonl(cache_path, [{"a": a, "b": b, "p": p, "part": "val"} for (a, b), p in val_missing])
    have = load_jsonl(cache_path)
    fit_needed = needed_inputs(parts["fit"])
    todo = [x for x in fit_needed if key_of(*x) not in have]
    if SMOKE:
        todo = todo[:SMOKE_INPUTS]
    for part in chunks(sorted(todo, key=lambda x: -len(x[0]) - len(x[1])), 64):
        res = backend.score(part)
        append_jsonl(cache_path, [{"a": a, "b": b, "p": r["p"], "mass": r["mass"], "part": "fit"} for (a, b), r in zip(part, res)])
    # 完成前逐项检查：没有重复、分区标对、概率有效、两部分都齐
    lines = [json.loads(line) for line in open(cache_path, encoding="utf-8")]
    keys = [key_of(x["a"], x["b"]) for x in lines]
    want = {key_of(*x): "val" for x in val_inputs}
    want.update({key_of(*x): "fit" for x in fit_needed})  # 同一输入两边都有时算作 fit（不会发生：两部分家族不相交）
    problems = []
    if len(keys) != len(set(keys)):
        problems.append("有重复的输入")
    if any(want.get(k) != x["part"] for k, x in zip(keys, lines)):
        problems.append("有输入的分区标错了或不属于这两部分")
    if not all(valid_prob(x["p"]) for x in lines):
        problems.append("有无效的概率")
    have_keys = set(keys)
    val_cov = sum(key_of(*x) in have_keys for x in val_inputs)
    fit_cov = sum(key_of(*x) in have_keys for x in fit_needed)
    if val_cov != len(val_inputs) or (not SMOKE and fit_cov != len(fit_needed)):
        problems.append(f"没导全：内层验证 {val_cov} / {len(val_inputs)}，训练部分 {fit_cov} / {len(fit_needed)}")
    if problems:
        raise SystemExit(f"seed{seed} 的导出有问题：{'；'.join(problems)}")
    by_key = dict(zip(keys, lines))
    # 统计按原始的 2,190 行训练记录（保留重复），学生训练时也按这些行映射
    rows = [r for r in parts["fit"] if key_of(r["text1"], r["text2"]) in by_key]
    probs = [by_key[key_of(r["text1"], r["text2"])]["p"] for r in rows]
    top = lambda p: max(range(4), key=lambda i: p[i])
    soft = lambda p, T: (lambda w: [x / sum(w) for x in w])([x ** (1 / T) for x in p])
    mx = sorted(max(p) for p in probs)
    mx2 = sorted(max(soft(p, 2)) for p in probs)
    stats = {"fit_rows": len(parts["fit"]), "fit_rows_with_teacher": len(rows), "fit_inputs": len(fit_needed), "best_epoch": best,
             "teacher_accuracy_on_fit_rows": round(sum(top(p) == LABELS.index(r["label"]) for p, r in zip(probs, rows)) / max(1, len(rows)), 4),
             "max_prob_median": round(statistics.median(mx), 4) if mx else None,
             "max_prob_above_0.999": round(sum(m > 0.999 for m in mx) / max(1, len(mx)), 4),
             "max_prob_median_T2": round(statistics.median(mx2), 4) if mx2 else None,
             "mean_entropy": round(sum(-sum(x * math.log(x) for x in p if x > 0) for p in probs) / max(1, len(probs)), 4),
             "zero_probs": sum(x == 0.0 for p in probs for x in p)}
    write_json(os.path.join(d, "done.json"), {"adapter_sha256": adapter_sha, "checks": checks, "stats": stats,
                                              "load_memory": backend.load_memory, "minutes": round((time.time() - t0) / 60, 1)})
    print(f"完成：seed{seed}  训练部分 {len(rows)} 行都有老师概率；老师在这些行上的准确率 {stats['teacher_accuracy_on_fit_rows']:.1%}，"
          f"最大概率中位数 {stats['max_prob_median']}（T = 2 后 {stats['max_prob_median_T2']}），"
          f"超过 0.999 的占 {stats['max_prob_above_0.999']:.1%}；用时 {(time.time() - t0) / 60:.1f} 分钟")


def pack():
    zip_path = os.path.join(WORK, f"{os.path.basename(OUT)}_{RUN_STAMP}.zip")
    with zipfile.ZipFile(zip_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)):
            if os.path.isfile(path) and not path.endswith(".tmp") and ".broken-" not in path:
                z.write(path, os.path.relpath(path, WORK))
    os.replace(zip_path + ".tmp", zip_path)
    return zip_path


def code_sha256():
    try:
        cells = [c for c in get_ipython().user_ns.get("In", []) if f'VERSION = "{VERSION}"' in c]  # noqa: F821
        if cells:
            return hashlib.sha256(cells[-1].encode("utf-8")).hexdigest(), "notebook cell"
    except NameError:
        pass
    path = globals().get("__file__")
    return (sha256(path), "file") if path and os.path.isfile(path) else (None, None)


def main():
    ensure_packages()
    ensure_official_repo()
    src, src_fp, adapter_shas = find_source()
    split = make_split()
    parts = track1_parts(split)
    assert (len(parts["fit"]), len(parts["val"]), len(parts["test"])) == (2190, 244, 608), "划分和第四组不同"
    check_data_and_split(src, src_fp, parts)
    os.makedirs(OUT, exist_ok=True)
    fp = {"version": VERSION, "source_fingerprint_sha256": SOURCE_FINGERPRINT_SHA256, "source_code_sha256": SOURCE_CODE_SHA256,
          "adapter_sha256": {str(s): h for s, h in adapter_shas.items()}, "smoke": SMOKE}
    fp_path = os.path.join(OUT, "fingerprint.json")
    if os.path.exists(fp_path) and read_json(fp_path) != fp:
        raise SystemExit(f"{OUT} 是另一份设置留下的，请删除或改名后重新运行")
    write_json(fp_path, fp)
    ok = False
    try:
        backend = QwenSFT(SOURCE_REVISION)
        code_hash, code_source = code_sha256()
        meta_path = os.path.join(OUT, "run_meta.json")
        runs = read_json(meta_path) if os.path.exists(meta_path) else []
        write_json(meta_path, runs + [{"start": RUN_STAMP, "version": VERSION, "code_sha256": code_hash,
                                       "code_source": code_source, "python": platform.python_version(), **backend.info()}])
        print("代码哈希：", code_hash, f"（{code_source}）")
        for seed in SEEDS:
            export_seed(backend, src, src_fp, seed, parts, adapter_shas[seed])
            pack()
        ok = True
    finally:
        zip_path = pack()
        print("\n结果包已保存到：", zip_path)
        print("全部完成，请把这个 zip 发给 Claude。" if ok else "运行中途停下了，报错信息在下面；重新运行会接着跑。")


if __name__ == "__main__":
    main()
