# %% [markdown]
# # DiCo-NLI 第四组实验：Qwen3-4B 的 LoRA 指令微调（SFT）
#
# 只有两个代码单元：先运行配置单元，再运行第二个单元（定义 + 训练 + 评分 + 打包），也可以直接 Run All。
#
# 对应导师清单第 1 项里的 "qwen"。模型和提示词都和第三组相同：`Qwen/Qwen3-4B-Instruct-2507`，第三组的 `zero` 提示词，在模型回答的开头替它写上 "Answer:"。区别是这里用训练集做 LoRA 微调：只在 "Answer:" 后面那一个答案字母上算交叉熵，提示词部分不计损失（和 IMDB 的 Qwen3-4B 实验相同）。
#
# **数据划分（和第二组 notebook 完全相同）**：英语训练集里文本相同的 pair_id 先合成家族，按家族分 5 折（种子 2027）。留出 fold 0（608 行）；fold 1–4 里再按家族分出 10% 作内层验证（244 行），其余 2,190 行用来训练。第二组 mDeBERTa 12 轮配方 fold 0 的折外模型用的是同一批训练、验证和留出数据，所以 fold 0 上可以直接比较。西班牙语、巴斯克语和跨语言赛道用同样的 pair_id 划分。
#
# **训练**：LoRA 加在 q/k/v/o/gate/up/down 七个投影层（r = 16，α = 32），学习率 1e-4，有效 batch 16，前 5% 步数线性升温，冻结的底座用 fp16、LoRA 参数用 fp32，开梯度检查点；这些都沿用 IMDB。最多训练 3 轮，每轮结束在内层验证上算原方向（plain）WF1，取最高的一轮（同分取早的），和 mDeBERTa 的选轮方式相同；比较时报 sym，也就是"按 plain 选轮后评 sym"。每个赛道 3 个种子（42、43、44）。不加类别权重：训练部分 2,190 行里 E / F / B / O 是 574 / 631 / 631 / 354，O 最少（16%）。
#
# **评测**：
#
# - 在 fold 0 和官方 dev 上评测；英语 fold 0 用来和 mDeBERTa 比较，dev 只报告。两种预测：`plain` 只看 (text1, text2)；`sym` 把对调输入的 F / B 概率互换，再和原方向取平均。相同的有序输入只算一次。
# - 每个赛道先用不加 LoRA 的同一个模型跑一次零样本，作为同一模型版本下"训练前"的对照。dev 上的零样本应和第三组的 `zero` 基本一致（英语对称化 WF1 .584），可以用来检查提示词和读概率没有写错。
# - mDeBERTa 的 dev 模型比这里多用了 fold 0 的数据训练（2,737 行，这里是 2,190 行），两边 dev 上的训练数据量不同；要据此做决定的比较看 fold 0。
# - dev 和训练集有少量文本重合：规范化（小写、合并空白、不计两侧顺序）后，dev 里 15 行（8 个 pair）的文本在训练集里出现过，其中 13 行（7 个 pair）在这里的训练部分。fold 0 按家族留出，没有这个问题。dev 照常报官方全量分数，并注明这一点。
#
# **事先定好的蒸馏条件**（由 Claude 用保存的结果分析，不在本 notebook 里；在看正式分数之前定下）：英语 fold 0 上满足任意一条，才试做"把大模型蒸馏给 mDeBERTa"：
#
# 1. 每个种子分别算 Qwen SFT 和 mDeBERTa 12 轮配方的 sym WF1，先求同一种子的差，再对 42 / 43 / 44 三个差取平均；平均差大于 0，而且按家族配对重抽样的近似 95% 区间下限大于 0。
# 2. 同一种子、同一有序输入，两个模型各自归一化的四类概率按 0.5 / 0.5 平均，再对称化、取最大、评分；"混合 − mDeBERTa"同样按种子求差再平均，下限也大于 0。
#
# 重抽样时每次抽同一批家族，三个种子都在这批家族上重算差值再平均；行、正反方向和三次训练都不当成独立样本。这个区间只反映"给定这三次训练结果"时评测样本的不确定性，不包含训练的随机性；两条规则"任一通过"也没有做多重比较校正，所以它只是决定要不要投入下一步的探索性门槛。通过了也只说明值得试：蒸馏实验要和同数据、同训练预算、不加蒸馏的 mDeBERTa 对照。fold 0 用来选路线以后，就不再当独立验证，之后的确认比较要换别的折。都不满足就不做蒸馏，接着做 mDeBERTa 上的 R-Drop、SCL、MoCo。
#
# **和 IMDB 原脚本的区别**：任务是四选一（四个字母）；多了每轮在内层验证上选轮；先用 `huggingface_hub.snapshot_download` 取得模型的 commit 号，再从这个快照加载，续跑时用同一个版本。
#
# **断点续跑**：训练每轮结束存一个检查点，被打断后从最近一份完整的检查点接着训练（写到一半的检查点会被移到 `checkpoints_invalid/` 并说明原因）；读概率每 64 个输入写一次缓存。训练完载入选中那一轮的 LoRA 权重时，逐个核对张量和保存的文件完全相同，并在内层验证上重评，和训练时保存的逐条概率比较。运行的代码哈希记在 `run_meta.json` 里，代码变了就不在同一个结果目录里接着跑。每个 (赛道, 种子) 跑完写 `done.json`，之后会跳过。每完成一部分就更新结果包 `dico_qwen_sft_results_<本次开始时间>.zip`。结果包里只放跑完的部分，不放 LoRA 权重和检查点（太大），所以换平台时，正在训练的那个种子要重新训练。在 Colab 上结果直接写在 Google Drive 里，断开后重新运行就能接着跑。
#
# **时间估计（Kaggle T4 上 SMOKE 实测）**：训练每步 5.7 秒（16 条、平均约 200 个 token），峰值显存 8.7 GiB。英语每个种子训练 3 轮（411 步）约 40 分钟，读概率约 3 分钟，加上加载模型，约 45 分钟；3 个种子加零样本约 2.5 小时。西班牙语、巴斯克语差不多；跨语言的训练集是 6 倍，每个种子约 4 小时。默认只跑英语（`TRACKS = [1]`），看过结果再加。
#
# **SMOKE（默认打开）**：每个分区按 pair 整对抽一个四类都有的小集合（约 192 条训练、32 条内层验证、fold 0 和 dev 各约 48 条，官方评分脚本照常运行），2 个种子、各 2 轮，十几分钟（另加下载模型）。除了走通全流程，还实测正式运行里的几种边界：
#
# - 第 1 个种子在第 2 轮中途**模拟一次中断**，再放一个写坏的"更新的"检查点，确认会跳过它、从第 1 轮的检查点恢复；
# - 训练完**逐轮载入**每一轮的 LoRA 权重，核对张量，并和训练时保存的内层验证概率比较；
# - 第 2 个种子**重新加载模型**，记录加载前后的显存，确认上一个模型已经释放。
#
# 最后打印这些检查是否通过，以及峰值显存、每步用时和正式运行的估计时间。请先跑 SMOKE，把输出和结果包发给 Claude 看过，再把 `SMOKE` 改成 `False` 正式运行。
#
# 在 Kaggle 上要打开 Internet（克隆官方仓库、下载约 8 GB 的模型），GPU 选 T4（T4 ×2 时只用第一张）。Cloud Studio 上如果下载不了 Hugging Face 的模型，先在第一个单元里加一行 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`。

# %%
import os
import sys
from pathlib import Path

if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
    raise RuntimeError("CUDA 已初始化。请重启会话（Colab：代码执行程序 → 重启会话；Kaggle：Restart Session），再从头运行。")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle 的 T4 ×2 只用第一张，避免 Trainer 自动多卡
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SMOKE = True  # 先跑 SMOKE；确认没问题后改成 False
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TRACKS = [1]  # 先跑英语；之后改成 [2, 3]，最后 [4]
SEEDS = [42, 43, 44]
HOLDOUT_FOLD = 0
MAX_EPOCHS = 3
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN_BATCH, GRAD_ACC = 8, 2  # 有效 batch 16，和 IMDB 相同
LEARNING_RATE = 1e-4
WARMUP = 0.05
WEIGHT_DECAY = 0.01
EVAL_TOKENS_PER_BATCH = 8192  # 读概率时一批最多多少个 token
SMOKE_ROWS = {"fit": 192, "val": 32, "test": 48, "dev": 48}

# Kaggle 的镜像里也有 google.colab 和 COLAB_RELEASE_TAG，所以先认 Kaggle；Colab 用它真机才有的 /var/colab/hostname 判断
ON_KAGGLE = Path("/kaggle/working").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = not ON_KAGGLE and Path("/var/colab/hostname").exists()
WORK = None
if "DICO_WORK" in os.environ:
    WORK = os.path.abspath(os.environ["DICO_WORK"])
elif ON_KAGGLE:
    WORK = "/kaggle/working"
elif ON_COLAB:  # Colab 断开后 /content 会清空，结果放在 Google Drive
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = "/content/drive/MyDrive/dico_qwen_sft"
    except Exception as e:
        print("挂载 Google Drive 失败，结果先写在 /content（断开后会丢，请及时下载结果包）：", e)
        WORK = "/content"
if WORK is None:
    WORK = os.path.abspath(".")
print("运行平台：", "Kaggle" if ON_KAGGLE else "Colab" if ON_COLAB else "其他 Linux", "；结果目录：", WORK)
os.makedirs(WORK, exist_ok=True)
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
OUT = os.path.join(WORK, "dico_qwen_sft_smoke" if SMOKE else "dico_qwen_sft_results")  # SMOKE 的结果单独放

FULL_EPOCHS = MAX_EPOCHS  # SMOKE 估计正式运行的时间时用
if SMOKE:
    SEEDS, MAX_EPOCHS = SEEDS[:2], 2

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
import subprocess
import time
import zipfile
from collections import Counter

VERSION = "dico_qwen_sft-2"
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


def build_prompt(a, b):
    return "\n\n".join([HEADER, "### Instruction:\n" + TASK + "\n" + ANSWER_LETTER,
                        f"### Input:\nPhrase 1: {a}\nPhrase 2: {b}", "### Response:"])


# ---------- 环境、数据 ----------

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


def smoke_pairs(rows, n):
    """SMOKE 用：在这个分区里按 pair 整对抽样（官方评分脚本要求的 reverse_pair_id 都在子集里），
    先保证四类各有一对，再按固定种子打乱的顺序补到约 n 行。"""
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    order = sorted(by_pair)
    random.Random(FOLD_SEED).shuffle(order)
    keep, count = [], 0
    for label in LABELS:
        p = next((p for p in order if p not in keep and any(r["label"] == label for r in by_pair[p])), None)
        if p is not None:
            keep.append(p)
            count += len(by_pair[p])
    for p in order:
        if count >= n:
            break
        if p not in keep:
            keep.append(p)
            count += len(by_pair[p])
    keep = set(keep)
    return [r for r in rows if r["pair_id"] in keep]


def track_parts(track, split, smoke):
    fam_of, fold_of, inner = split
    parts = {"fit": [], "val": [], "test": []}
    for r in read_rows(data_file(track, "train", "participant_labeled")):
        f = fam_of[r["pair_id"]]
        parts["test" if fold_of[f] == HOLDOUT_FOLD else "val" if f in inner else "fit"].append(r)
    parts["dev"] = read_rows(data_file(track, "dev", "participant_labeled"))
    if smoke:
        parts = {k: smoke_pairs(v, SMOKE_ROWS[k]) for k, v in parts.items()}
    return parts


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


# ---------- 指纹、结果包 ----------

def fingerprint():
    files = sorted({data_file(t, s, k) for t in (1, 2, 3, 4) for s in ("train", "dev")
                    for k in ("participant_labeled", "reference")})
    return {
        "version": VERSION, "model": MODEL, "pins": PINS,
        "prompt_text": {"header": HEADER, "task": TASK, "answer_letter": ANSWER_LETTER, "marker": MARKER},
        "split": {"folds": FOLDS, "fold_seed": FOLD_SEED, "inner_fraction": INNER_FRACTION, "holdout_fold": HOLDOUT_FOLD},
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
        "train": {"max_epochs": MAX_EPOCHS, "batch": TRAIN_BATCH, "grad_acc": GRAD_ACC, "lr": LEARNING_RATE,
                  "warmup": WARMUP, "weight_decay": WEIGHT_DECAY},
        "smoke": SMOKE, "smoke_rows": SMOKE_ROWS if SMOKE else None,
        "data_sha256": {os.path.relpath(p, DICO_DIR): sha256(p) for p in files},
    }


def incompatible(old, fp):
    return sorted(k for k in set(fp) | set(old) if k != "model_revision" and fp.get(k) != old.get(k))


def restore_previous_results(fp):
    """只从设置相同的结果包恢复；有多个时选完成的部分最多、其次缓存条数最多的。"""
    if SMOKE or os.path.isdir(OUT):
        return
    base = os.path.basename(OUT)
    found = sorted(set(glob.glob(os.path.join(WORK, f"{base}_*.zip"))
                       + glob.glob(f"/kaggle/input/**/{base}_*.zip", recursive=True)))
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
        print(f"已从结果包恢复（完成 {best_progress[0]} 项，缓存 {best_progress[1]} 条）：", best)


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


def code_sha256():
    """这次实际运行的代码：notebook 里取含 VERSION 的这个单元的源码（配置单元的取值已在指纹里），当脚本运行时取文件。"""
    try:
        cells = [c for c in get_ipython().user_ns.get("In", []) if f'VERSION = "{VERSION}"' in c]  # noqa: F821
        if cells:
            return hashlib.sha256(cells[-1].encode("utf-8")).hexdigest(), "notebook cell"
    except NameError:
        pass
    path = globals().get("__file__")
    if path and os.path.isfile(path):
        return sha256(path), "file"
    return None, None


def check_code(code_hash):
    """同一个结果目录里只用同一份代码。SMOKE 的目录可以随时重来，代码变了就清空。"""
    path = os.path.join(OUT, "run_meta.json")
    old = {r.get("code_sha256") for r in (read_json(path) if os.path.exists(path) else [])} - {None}
    if not old or old == {code_hash} or code_hash is None:
        return
    if SMOKE:
        print("代码和上次 SMOKE 不同，清空 SMOKE 的结果目录重来")
        fp = read_json(os.path.join(OUT, "fingerprint.json"))
        shutil.rmtree(OUT)
        os.makedirs(OUT)
        write_json(os.path.join(OUT, "fingerprint.json"), fp)
        return
    raise SystemExit(f"这个结果目录是用另一份代码（{sorted(old)}）跑的，现在的代码哈希是 {code_hash}。"
                     f"不把新旧代码的输出混在一起：要接着跑，请换回原来的 notebook；要重新开始，请删除或改名 {OUT}。")


def record_run(info):
    path = os.path.join(OUT, "run_meta.json")
    code_hash, code_source = code_sha256()
    check_code(code_hash)
    runs = read_json(path) if os.path.exists(path) else []
    runs.append({"start": RUN_STAMP, "version": VERSION, "code_sha256": code_hash, "code_source": code_source,
                 "python": platform.python_version(), "train_batch": TRAIN_BATCH,
                 "grad_acc": GRAD_ACC, "eval_tokens_per_batch": EVAL_TOKENS_PER_BATCH, **info})
    write_json(path, runs)
    print("代码哈希：", code_hash, f"（{code_source}）")


def packable(path):
    """结果包只放跑完的种子（没跑完的缺 LoRA 权重，换平台也接不上），也不放 LoRA 权重和检查点（每个几百 MB）。"""
    rel = os.path.relpath(path, OUT).split(os.sep)
    if path.endswith(".tmp") or ".broken-" in rel[-1] or {"adapters", "checkpoints", "checkpoints_invalid"} & set(rel):
        return False
    if len(rel) >= 3 and rel[0].startswith("track") and rel[1].startswith("seed"):
        return os.path.exists(os.path.join(OUT, rel[0], rel[1], "done.json"))
    return True


def pack():
    if not os.path.isdir(OUT):
        return None
    zip_path = os.path.join(WORK, f"{os.path.basename(OUT)}_{RUN_STAMP}.zip")
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)):
            if os.path.isfile(path) and packable(path):
                z.write(path, os.path.relpath(path, WORK))
    os.replace(tmp, zip_path)
    return zip_path


# ---------- 模型（唯一用到 torch 的部分） ----------

class SimulatedInterrupt(RuntimeError):
    """SMOKE 里模拟的中断。"""


def steps_per_epoch(n_rows):
    """每轮的参数更新次数：最后一个不满的累积组也算一步（和 Trainer 相同）。"""
    return math.ceil(math.ceil(n_rows / TRAIN_BATCH) / GRAD_ACC)


REQUIRED_CKPT_FILES = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors"]


def find_checkpoint(ckpt_dir, spe, fp16):
    """从新到旧找一份完整的检查点。Trainer 最后写 trainer_state.json，所以它能完整解析、轮数是整数、
    步数等于轮数 × 每轮步数，而且优化器、调度器、LoRA 权重都在，才算完整。
    不完整的移到 checkpoints_invalid/ 并说明原因（留着它，Trainer 轮换检查点时会把它当成最新的）。"""
    cands = []
    for path in glob.glob(os.path.join(ckpt_dir, "checkpoint-*")):
        tail = os.path.basename(path).split("-", 1)[1]
        if os.path.isdir(path) and tail.isdigit():
            cands.append((int(tail), path))
    skipped = []
    for num, path in sorted(cands, reverse=True):
        reason, state = None, None
        missing = [f for f in REQUIRED_CKPT_FILES if not os.path.exists(os.path.join(path, f))]
        if missing:
            reason = f"缺少 {missing}"
        else:
            try:
                state = read_json(os.path.join(path, "trainer_state.json"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                reason = f"trainer_state.json 不完整（{type(e).__name__}）"
        if state is not None:
            ep, gs = state.get("epoch"), state.get("global_step")
            if not isinstance(ep, (int, float)) or abs(ep - round(ep)) > 1e-6 or round(ep) < 1:
                reason = f"epoch = {ep!r} 不是完整的轮"
            elif gs != round(ep) * spe or gs != num:
                reason = f"global_step = {gs!r}，和第 {round(ep)} 轮（每轮 {spe} 步）或目录编号 {num} 对不上"
        if reason:
            dest = os.path.join(os.path.dirname(ckpt_dir), "checkpoints_invalid", f"{os.path.basename(path)}-{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(path, dest)
            skipped.append({"checkpoint": os.path.basename(path), "reason": reason})
            print(f"    跳过检查点 {os.path.basename(path)}：{reason}；已移到 {dest}", flush=True)
            continue
        optional = ["rng_state.pth"] + (["scaler.pt"] if fp16 else [])
        missing_optional = [f for f in optional if not os.path.exists(os.path.join(path, f))]
        if missing_optional:
            print(f"    注意：检查点 {os.path.basename(path)} 里没有 {missing_optional}，恢复时随机数或 fp16 缩放状态可能和中断前不同", flush=True)
        return {"path": path, "checkpoint": os.path.basename(path), "epoch": round(state["epoch"]),
                "files": sorted(os.listdir(path)), "missing_optional": missing_optional, "skipped": skipped}
    return {"path": None, "checkpoint": None, "epoch": 0, "files": [], "missing_optional": [], "skipped": skipped}


def plant_broken_checkpoint(run_dir):
    """SMOKE 用：放一个编号更大、但 trainer_state.json 只写了一半的检查点，看续跑会不会跳过它。"""
    d = os.path.join(run_dir, "checkpoints", "checkpoint-999999")
    os.makedirs(d, exist_ok=True)
    for name in REQUIRED_CKPT_FILES[1:]:
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"placeholder")
    with open(os.path.join(d, "trainer_state.json"), "w") as f:
        f.write('{"epoch":')
    return os.path.basename(d)


def best_of(history):
    return max(history, key=lambda h: (h["val_weighted_f1"], -h["epoch"]))["epoch"]  # 同分取早的


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

    def train(self, run_dir, fit_rows, val_rows, seed, interrupt_at=None):
        """LoRA 微调。每轮结束保存这一轮的 LoRA 权重和内层验证的逐条概率，按原方向 WF1 选轮；训练完载入选中的一轮并核对。
        interrupt_at 只在 SMOKE 里用：训练到这一步时模拟一次中断。"""
        torch = self.torch
        import torch.nn.functional as F
        from datasets import Dataset
        from peft import (LoraConfig, TaskType, cast_mixed_precision_params, get_peft_model, get_peft_model_state_dict,
                          set_peft_model_state_dict)
        from safetensors.torch import load_file
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

        backend = self
        hist_path, trained_path = os.path.join(run_dir, "history.json"), os.path.join(run_dir, "trained.json")
        adapter_dir = lambda ep: os.path.join(run_dir, "adapters", f"epoch{ep}")
        val_probs_path = lambda ep: os.path.join(run_dir, "val_probs", f"epoch{ep}.json")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if os.path.exists(trained_path) and not os.path.exists(adapter_dir(read_json(trained_path)["best_epoch"])):
            print("    找不到训练好的 LoRA 权重，这个种子重新训练", flush=True)
            for name in os.listdir(run_dir):  # 旧缓存是用旧权重算的，一起清掉
                path = os.path.join(run_dir, name)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

        self.load_base()
        set_seed(seed)  # LoRA 的初始权重由这个种子决定
        model = get_peft_model(self.model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                                                      lora_dropout=LORA_DROPOUT, bias="none", target_modules=LORA_TARGETS))
        cast_mixed_precision_params(model, dtype=self.dtype)  # 冻结的底座半精度，LoRA 参数 fp32
        self.model = model
        val_inputs = list(dict.fromkeys((r["text1"], r["text2"]) for r in val_rows))

        def val_probs():
            return [x["p"] for x in backend.score(val_inputs)]

        def f1_of(probs):
            by = dict(zip(val_inputs, probs))
            return weighted_f1([r["label"] for r in val_rows], [argmax_label(by[(r["text1"], r["text2"])]) for r in val_rows])

        def load_epoch(ep, history):
            """载入第 ep 轮的 LoRA 权重并核对：文件里的 key 和形状要和模型的 LoRA 参数一一对应，载入后每个张量和文件逐位相同；
            再在内层验证上重评，和这一轮训练时保存的逐条概率比较。"""
            saved = load_file(os.path.join(adapter_dir(ep), "adapter_model.safetensors"))
            expected = get_peft_model_state_dict(model)
            missing, extra = sorted(set(expected) - set(saved)), sorted(set(saved) - set(expected))
            shapes = [k for k in saved if k in expected and tuple(saved[k].shape) != tuple(expected[k].shape)]
            if missing or extra or shapes:
                raise SystemExit(f"第 {ep} 轮的 LoRA 文件和模型对不上：缺 {missing[:3]}，多 {extra[:3]}，形状不同 {shapes[:3]}")
            res = set_peft_model_state_dict(model, saved)
            if getattr(res, "unexpected_keys", None):
                raise SystemExit(f"载入第 {ep} 轮的 LoRA 权重时有对不上的 key：{res.unexpected_keys[:5]}")
            now = get_peft_model_state_dict(model)
            differ = [k for k in saved if not torch.equal(now[k].detach().cpu().to(saved[k].dtype), saved[k])]
            if differ:
                raise SystemExit(f"载入第 {ep} 轮后有 {len(differ)} 个张量和文件不同，例如 {differ[:3]}")
            probs, old = val_probs(), read_json(val_probs_path(ep))
            top = lambda p: max(range(4), key=lambda i: p[i])
            agree = sum(top(p) == top(q) for p, q in zip(probs, old)) / len(old)
            diff = max(abs(a - b) for p, q in zip(probs, old) for a, b in zip(p, q))
            check = {"epoch": ep, "tensors": len(saved), "val_argmax_agreement": round(agree, 4),
                     "val_max_prob_diff": round(diff, 6), "val_weighted_f1": f1_of(probs),
                     "recorded_val_weighted_f1": [h for h in history if h["epoch"] == ep][0]["val_weighted_f1"]}
            print(f"    载入第 {ep} 轮的权重：{len(saved)} 个张量和文件逐位相同；内层验证逐条预测和训练时一致 {agree:.1%}，"
                  f"概率最大差 {diff:.2g}，WF1 {check['val_weighted_f1']:.4f}（训练时 {check['recorded_val_weighted_f1']:.4f}）", flush=True)
            if agree < 0.98:
                raise SystemExit(f"第 {ep} 轮的权重载入后，内层验证有 {1 - agree:.1%} 的预测和训练时不同，请把输出发给 Claude")
            return check

        def finish(history):
            """SMOKE 里逐轮载入核对，最后都换回选中的一轮。"""
            best = best_of(history)
            checks = [load_epoch(h["epoch"], history) for h in history if SMOKE and h["epoch"] != best]
            return best, checks + [load_epoch(best, history)]

        if os.path.exists(trained_path):
            history = read_json(hist_path)
            finish(history)
            return history, read_json(trained_path)

        enc = {"input_ids": [], "answer_id": [], "length": []}
        for r in fit_rows:
            ids = self.encode(r["text1"], r["text2"])
            enc["input_ids"].append(ids)
            enc["answer_id"].append(self.target_ids[LABELS.index(r["label"])])
            enc["length"].append(len(ids))

        resume = find_checkpoint(ckpt_dir, steps_per_epoch(len(fit_rows)), fp16=not self.bf16)
        done_epochs = resume["epoch"]
        history = [h for h in read_json(hist_path) if h["epoch"] <= done_epochs] if done_epochs and os.path.exists(hist_path) else []
        if sorted(h["epoch"] for h in history) != list(range(1, done_epochs + 1)):
            raise SystemExit(f"{hist_path} 和检查点 {resume['checkpoint']} 对不上。请把输出发给 Claude。")
        if done_epochs:
            interrupt_at = None
            print(f"    从检查点 {resume['checkpoint']} 接着训练（已完成 {done_epochs} 轮）", flush=True)
        else:
            print("    从头训练", flush=True)

        class LastTokenTrainer(Trainer):
            # 损失 = 答案 token 在整个词表上的交叉熵，提示词部分不计损失（和 IMDB 相同）
            loss_is_scaled_for_ga = False

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                logits = backend._last_logits(model, inputs["input_ids"], inputs["attention_mask"])
                loss = F.cross_entropy(logits.float(), inputs["answer_id"])
                return (loss, logits) if return_outputs else loss

        def collate(rows):
            ids, mask = backend._pad([r["input_ids"] for r in rows])
            return {"input_ids": ids, "attention_mask": mask,
                    "answer_id": torch.tensor([r["answer_id"] for r in rows], dtype=torch.long)}

        class EpochEval(TrainerCallback):
            """在 Trainer 保存这一轮的检查点之前运行；被打断后重跑这一轮，会覆盖同一轮的记录、权重和概率。"""

            def on_epoch_begin(self, args, state, control, **kwargs):
                self.t0 = time.time()

            def on_epoch_end(self, args, state, control, **kwargs):
                ep = int(round(state.epoch))
                model.save_pretrained(adapter_dir(ep))
                probs = val_probs()
                write_json(val_probs_path(ep), probs)
                f1 = f1_of(probs)
                losses = [h["loss"] for h in state.log_history if "loss" in h and ep - 1 < h.get("epoch", 0) <= ep]
                history[:] = sorted([h for h in history if h["epoch"] != ep] + [{
                    "epoch": ep, "train_loss": sum(losses) / len(losses) if losses else None,
                    "val_weighted_f1": f1, "minutes": round((time.time() - self.t0) / 60, 1)}], key=lambda h: h["epoch"])
                write_json(hist_path, history)
                loss = history[-1]["train_loss"]
                print(f"    第 {ep} 轮：训练损失 {'-' if loss is None else f'{loss:.4f}'}  内层验证 WF1 {f1:.4f}"
                      f"（目前最好是第 {best_of(history)} 轮）", flush=True)

        class Progress(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                self.start, self.marks = state.global_step, {}

            def on_step_end(self, args, state, control, **kwargs):
                self.marks[state.global_step] = time.time()
                ran = state.global_step - self.start
                if ran in (5, 20) or state.global_step % 50 == 0 or state.global_step == state.max_steps:
                    first = min(self.marks)
                    rate = (time.time() - self.marks[first]) / max(1, state.global_step - first)
                    print(f"    step {state.global_step}/{state.max_steps}  {rate:.1f} 秒/步  "
                          f"这一组的训练预计还要 {(state.max_steps - state.global_step) * rate / 60:.0f} 分钟", flush=True)
                if interrupt_at is not None and state.global_step == interrupt_at:
                    raise SimulatedInterrupt(f"SMOKE：在第 {state.global_step} 步模拟中断")

        args = TrainingArguments(
            output_dir=ckpt_dir, num_train_epochs=MAX_EPOCHS, per_device_train_batch_size=TRAIN_BATCH,
            gradient_accumulation_steps=GRAD_ACC, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP, lr_scheduler_type="linear", optim="adamw_torch", fp16=not self.bf16, bf16=self.bf16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            train_sampling_strategy="group_by_length", length_column_name="length", remove_unused_columns=False,
            label_names=["answer_id"], eval_strategy="no", save_strategy="epoch", save_total_limit=1,
            logging_steps=10, seed=seed, data_seed=seed, dataloader_num_workers=0, report_to="none")
        progress = Progress()
        trainer = LastTokenTrainer(model=model, args=args, train_dataset=Dataset.from_dict(enc), data_collator=collate,
                                   processing_class=self.tok, callbacks=[EpochEval(), progress])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train(resume_from_checkpoint=resume["path"])
        steps = sorted(progress.marks)
        best, checks = finish(history)
        info = {"best_epoch": best, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "steps": trainer.state.global_step, "steps_per_epoch": steps_per_epoch(len(fit_rows)),
                "train_minutes_last_session": round((time.time() - t0) / 60, 1),
                # 从第 2 步算起的平均每步用时，不含第一步的预热
                "seconds_per_step": round((progress.marks[steps[-1]] - progress.marks[steps[1]]) / (steps[-1] - steps[1]), 2)
                if len(steps) > 2 else None,
                "resumed_from_epoch": done_epochs, "resume": {k: v for k, v in resume.items() if k != "path"},
                "adapter_checks": checks, "load_memory": self.load_memory,
                "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                "fit_rows": len(fit_rows), "val_rows": len(val_rows), "fit_tokens": sum(enc["length"])}
        write_json(trained_path, info)
        del trainer
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        for ep in range(1, MAX_EPOCHS + 1):  # 只留选中那一轮的 LoRA 权重；每轮的内层验证概率都留着
            if ep != best:
                shutil.rmtree(adapter_dir(ep), ignore_errors=True)
        return history, info


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
    res = subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", gold,
                          "--predictions", pred, "--output-dir", out_dir],
                         cwd=DICO_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"官方评分脚本出错（{os.path.basename(pred)}）：{(res.stdout + res.stderr).strip()[-800:]}")
    s = read_json(os.path.join(out_dir, "scores.json"))
    return {k: s[k] for k in METRICS}


def reference_for(track, split_name, rows, d, name):
    """评分用的金标文件；只评一部分行（fold 0、SMOKE）时写一个子集。fold 0 按家族整体留出，正反两个方向都在里面。
    子集只写评分脚本要的四列：脚本用 csv.Sniffer 猜格式，短语里的撇号（如 "'s war"）会被它当成引号。"""
    path = data_file(track, split_name, "reference")
    ref_rows = read_rows(path)
    keep = {r["instance_id"] for r in rows}
    if len(keep) == len(ref_rows):
        return path
    sub = [{k: r[k] for k in REFERENCE_FIELDS} for r in ref_rows if r["instance_id"] in keep]
    assert len(sub) == len(keep), "评分的行在金标文件里找不到"
    out = os.path.join(d, f"reference_{name}.csv")
    write_rows(out, sub, REFERENCE_FIELDS)
    return out


def score_and_evaluate(backend, d, track, parts):
    """fold 0 和 dev 的全部有序输入读概率（写缓存，可续跑），再出 plain / sym 两种预测并用官方脚本评分。"""
    cache_path = os.path.join(d, "cache.jsonl")
    cache = load_jsonl(cache_path)
    inputs = needed_inputs(parts["test"] + parts["dev"])
    todo = [x for x in inputs if key_of(*x) not in cache]
    t0 = time.time()
    for part in chunks(sorted(todo, key=lambda x: -len(x[0]) - len(x[1])), 64):
        res = backend.score(part)
        items = [{"a": a, "b": b, **r} for (a, b), r in zip(part, res)]
        append_jsonl(cache_path, items)
        cache.update({key_of(x["a"], x["b"]): x for x in items})
    prob = lambda r, swap=False: cache[key_of(r["text2"], r["text1"]) if swap else key_of(r["text1"], r["text2"])]["p"]
    scores = {}
    for name, split_name, rows in (("fold0", "train", parts["test"]), ("dev", "dev", parts["dev"])):
        ref = reference_for(track, split_name, rows, d, name)
        scores[name] = {}
        for mode in ("plain", "sym"):
            labels = [argmax_label(prob(r) if mode == "plain" else sym(prob(r), prob(r, True))) for r in rows]
            path = os.path.join(d, f"pred_{name}_{mode}.csv")
            write_rows(path, [{"instance_id": r["instance_id"], "label": l} for r, l in zip(rows, labels)],
                       ["instance_id", "label"])
            scores[name][mode] = official_score(ref, path, os.path.join(d, f"scores_{name}_{mode}"))
    masses = sorted(cache[key_of(*x)]["mass"] for x in inputs)
    stats = {"inputs": len(inputs), "scored_this_session": len(todo),
             "score_seconds_this_session": round(time.time() - t0, 1),
             "mean_answer_mass": round(sum(masses) / len(masses), 4),
             "answer_mass_p05": round(masses[int(0.05 * len(masses))], 4),
             "answer_mass_below_0.5": sum(m < 0.5 for m in masses),
             "top_is_letter": round(sum(cache[key_of(*x)]["top_is_letter"] for x in inputs) / len(inputs), 4)}
    return scores, stats


def fmt(s):
    return " / ".join(f"{s[k]:.3f}" for k in METRICS)


def run_zero_shot(backend, track, parts):
    d = os.path.join(OUT, f"track{track}", "zero_shot")
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"跳过（已完成）：track{track} 零样本")
        return
    t0 = time.time()
    print(f"开始：track{track} 零样本（同一模型版本、不加 LoRA）", flush=True)
    backend.load_base()
    scores, stats = score_and_evaluate(backend, d, track, parts)
    write_json(os.path.join(d, "done.json"), {"scores": scores, "stats": stats, "load_memory": backend.load_memory,
                                              "minutes": round((time.time() - t0) / 60, 1)})
    print(f"完成：track{track} 零样本  fold 0 sym {fmt(scores['fold0']['sym'])}  dev sym {fmt(scores['dev']['sym'])}"
          f"  用时 {(time.time() - t0) / 60:.1f} 分钟")


def run_seed(backend, track, seed, parts):
    d = os.path.join(OUT, f"track{track}", f"seed{seed}")
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"跳过（已完成）：track{track} seed{seed}")
        return
    t0 = time.time()
    print(f"开始：track{track} seed{seed}：训练 {len(parts['fit'])} 行，内层验证 {len(parts['val'])} 行，"
          f"fold 0 {len(parts['test'])} 行，dev {len(parts['dev'])} 行", flush=True)
    os.makedirs(d, exist_ok=True)
    interrupt_at = None
    fresh = not os.path.exists(os.path.join(d, "checkpoints")) and not os.path.exists(os.path.join(d, "trained.json"))
    if SMOKE and seed == SEEDS[0] and fresh and MAX_EPOCHS >= 2:  # SMOKE：第 2 轮的第 2 步模拟中断
        interrupt_at = steps_per_epoch(len(parts["fit"])) + 2
    interrupted = False
    try:
        history, info = backend.train(d, parts["fit"], parts["val"], seed, interrupt_at)
    except SimulatedInterrupt as e:
        interrupted = str(e)
    if interrupted:  # 在 except 外面接着跑，让异常连同它引用的旧模型先释放
        broken = plant_broken_checkpoint(d)
        print(f"  {interrupted}；另放了一个写坏的检查点 {broken}，接着从检查点恢复", flush=True)
        backend.release()
        history, info = backend.train(d, parts["fit"], parts["val"], seed)
    scores, stats = score_and_evaluate(backend, d, track, parts)
    write_json(os.path.join(d, "done.json"), {"scores": scores, "stats": stats, "history": history, **info,
                                              "minutes_last_session": round((time.time() - t0) / 60, 1)})
    print(f"完成：track{track} seed{seed}  选中第 {info['best_epoch']} 轮  fold 0 sym {fmt(scores['fold0']['sym'])}"
          f"  dev sym {fmt(scores['dev']['sym'])}  用时 {(time.time() - t0) / 60:.1f} 分钟")


def summarize():
    out, missing = {}, []
    for track in TRACKS:
        zs_path = os.path.join(OUT, f"track{track}", "zero_shot", "done.json")
        paths = {s: os.path.join(OUT, f"track{track}", f"seed{s}", "done.json") for s in SEEDS}
        if not os.path.exists(zs_path):
            missing.append(f"track{track} 零样本")
        missing += [f"track{track} seed{s}" for s, p in paths.items() if not os.path.exists(p)]
        done = {s: read_json(p) for s, p in paths.items() if os.path.exists(p)}
        t = {"zero_shot": read_json(zs_path)["scores"] if os.path.exists(zs_path) else None, "sft": {},
             "best_epochs": {s: d["best_epoch"] for s, d in done.items()}}
        for name in ("fold0", "dev"):
            for mode in ("plain", "sym"):
                for k in METRICS:
                    vals = [d["scores"][name][mode][k] for d in done.values()]
                    if vals:
                        mean = sum(vals) / len(vals)
                        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
                        t["sft"].setdefault(f"{name}_{mode}", {})[k] = {
                            "mean": round(mean, 4), "sd": None if sd is None else round(sd, 4), "n": len(vals)}
        out[f"track{track}"] = t
    write_json(os.path.join(OUT, "summary.json"), {"version": VERSION, "model": MODEL, "results": out, "missing": missing})
    for track, t in out.items():
        print(f"\n{track}（WF1 / SoftCons / HardCons；SFT 是 {len(t['best_epochs'])} 个种子的均值）")
        for name in ("fold0", "dev"):
            for mode in ("plain", "sym"):
                zs = fmt(t["zero_shot"][name][mode]) if t["zero_shot"] else "-"
                sft = t["sft"].get(f"{name}_{mode}")
                sft = " / ".join(f"{sft[k]['mean']:.3f}" for k in METRICS) if sft else "-"
                print(f"  {name:<5} {mode:<5}  零样本 {zs:<21}  SFT {sft}")
        print("  每个种子选中的轮：", t["best_epochs"] or "-")
    return missing


def smoke_report(backend, split, track):
    """SMOKE：逐项列出边界检查是否通过，并用实测的每步用时、峰值显存估计正式运行要多久。"""
    first, second = (read_json(os.path.join(OUT, f"track{track}", f"seed{s}", "done.json")) for s in SEEDS[:2])
    zs = read_json(os.path.join(OUT, f"track{track}", "zero_shot", "done.json"))
    parts = track_parts(track, split, smoke=True)
    spe = steps_per_epoch(len(parts["fit"]))
    checks = {
        "四个分区都含四类": all({r["label"] for r in parts[k]} == set(LABELS) for k in ("fit", "val", "test", "dev")),
        "中断后从第 1 轮的检查点恢复": first["resumed_from_epoch"] == 1 and first["resume"]["checkpoint"] == f"checkpoint-{spe}",
        "写坏的检查点被跳过": [x["checkpoint"] for x in first["resume"]["skipped"]] == ["checkpoint-999999"],
        "检查点里有 RNG 和 fp16 缩放状态": not first["resume"]["missing_optional"],
        "每一轮的权重载入后张量逐位相同、预测一致": all(
            {c["epoch"] for c in d["adapter_checks"]} == {h["epoch"] for h in d["history"]}
            and all(c["val_argmax_agreement"] >= 0.98 for c in d["adapter_checks"]) for d in (first, second)),
        "第二个种子加载前显存已释放": second["load_memory"]["before_load"]["allocated_gib"] < 0.5,
    }
    full = track_parts(track, split, smoke=False)
    full_tokens = sum(len(backend.encode(r["text1"], r["text2"])) for r in full["fit"])
    steps = steps_per_epoch(len(full["fit"])) * FULL_EPOCHS
    scale = (full_tokens / len(full["fit"])) / (second["fit_tokens"] / second["fit_rows"])  # 平均长度不同时按比例换算
    train_min = steps * (second["seconds_per_step"] or 0) * scale / 60
    per_input = zs["stats"]["score_seconds_this_session"] / max(1, zs["stats"]["scored_this_session"])
    n_scored = len(needed_inputs(full["test"] + full["dev"])) + len({(r["text1"], r["text2"]) for r in full["val"]}) * FULL_EPOCHS
    report = {"checks": checks, "labels": {k: dict(Counter(r["label"] for r in v)) for k, v in parts.items()},
              "checkpoint_files": first["resume"]["files"], "skipped_checkpoints": first["resume"]["skipped"],
              "adapter_checks": {f"seed{s}": d["adapter_checks"] for s, d in zip(SEEDS, (first, second))},
              "load_memory": {"zero_shot": zs["load_memory"], **{f"seed{s}": d["load_memory"] for s, d in zip(SEEDS, (first, second))}},
              "peak_memory_gib": max(first["peak_memory_gib"], second["peak_memory_gib"]),
              "seconds_per_step": second["seconds_per_step"], "full_fit_rows": len(full["fit"]), "full_fit_tokens": full_tokens,
              "estimated_train_minutes_per_seed": round(train_min),
              "estimated_scoring_minutes_per_seed": round(per_input * n_scored / 60),
              "smoke_scores": {"zero_shot": zs["scores"], **{f"seed{s}": d["scores"] for s, d in zip(SEEDS, (first, second))}}}
    write_json(os.path.join(OUT, "smoke_checks.json"), report)
    print("\nSMOKE 的边界检查：")
    for name, ok in checks.items():
        print(f"  {'通过  ' if ok else '没通过'}  {name}")
    print("  恢复时用的检查点里的文件：", first["resume"]["files"])
    print(f"\nSMOKE：训练峰值显存 {report['peak_memory_gib']} GiB，每步 {report['seconds_per_step']} 秒。"
          f"正式运行（{len(full['fit'])} 行，{FULL_EPOCHS} 轮）估计每个种子训练约 {report['estimated_train_minutes_per_seed']} 分钟、"
          f"读概率约 {report['estimated_scoring_minutes_per_seed']} 分钟，另加每次加载模型的一两分钟。")
    r = full["dev"][0]
    print("示例输入的结尾（模型在这之后读出字母概率）：", repr(backend.chat(build_prompt(r["text1"], r["text2"])) + MARKER)[-160:])
    return all(checks.values())


def main():
    ensure_packages()
    ensure_official_repo()
    fp = fingerprint()
    restore_previous_results(fp)
    os.makedirs(OUT, exist_ok=True)
    revision = check_fingerprint(fp)
    split = make_split()
    missing, smoke_ok = None, None
    try:
        backend = QwenSFT(revision)
        record_revision(backend.revision)
        record_run(backend.info())
        for track in TRACKS:
            parts = track_parts(track, split, SMOKE)
            run_zero_shot(backend, track, parts)
            pack()
            for seed in SEEDS:
                run_seed(backend, track, seed, parts)
                pack()
            if SMOKE:
                smoke_ok = smoke_report(backend, split, track)
        missing = summarize()
    finally:
        if missing is None:
            print("\n运行中途停下了，下面是已经完成的部分：")
            summarize()
        zip_path = pack()
        print("\n结果包已保存到：", zip_path)
        if missing is None:
            print("运行出错停下了，报错信息在下面。已完成的部分已保存，排除问题后重新运行本 notebook 会接着跑。")
        elif SMOKE:
            print("SMOKE 跑完了，边界检查" + ("全部通过。" if smoke_ok else "有没通过的项（见上面）。")
                  + "请把上面的输出和这个 zip 发给 Claude；确认没问题后，把第一个单元的 SMOKE 改成 False，重启会话再正式运行。")
        elif missing:
            print("还没跑完：", missing, "。重新运行本 notebook 会接着跑（换平台时把这个 zip 带过去）。")
        else:
            print("全部跑完了。请把这个 zip 下载下来发给 Claude。")


if __name__ == "__main__":
    main()
