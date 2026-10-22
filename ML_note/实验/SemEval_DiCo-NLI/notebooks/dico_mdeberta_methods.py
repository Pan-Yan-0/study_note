# %% [markdown]
# # DiCo-NLI 第五组实验：mDeBERTa 上的蒸馏、R-Drop、方向版 R-Drop、SCL、MoCo
#
# 只有两个代码单元：配置单元和主体单元。在阿里云 PAI-DSW 上用配套的 `dico_a10_mdeberta.sh` 在终端里跑 `.py`；在 Kaggle / Colab 上可以直接 Run All（先在配置单元里设 `STAGE`）。
#
# **做什么**：在第二组定下的 mDeBERTa 12 轮配方上，一次只加一种方法，和同一环境里重跑的只用交叉熵的基线（C0）按种子配对比较。所有方法的超参数事先固定，见配置单元；结论只说"在这组设置下"。
#
# | 编号 | 方法目录 | 损失 |
# |---|---|---|
# | C0 | `ce` | 交叉熵（和第二组 e12 相同） |
# | B2 | `kd` | 0.5 × 交叉熵 + 0.5 × T² × KL(老师 ‖ 学生)，T = 2；老师是第四组英语 3 轮 Qwen SFT 的同种子模型，只用原始的 2,190 行（保留重复） |
# | C1 | `rdrop` | 同一批输入前向两次：(CE₁ + CE₂) / 2 + α × 对称 KL，α = 1 |
# | C2 | `dir_rdrop` | 交叉熵 + α × 对称 KL(p(A, B), F/B 对调后的 p(B, A))，α = 1，只对金标是 E / F / B 的行；反向输入不算交叉熵 |
# | C3 | `scl` | 0.7 × 交叉熵 + 0.3 × 监督对比损失（τ = 0.3），用最后一层 [CLS] 加一个两层投影（投影是我们的选择） |
# | C4 | `moco_m0.99`、`moco_m0.999` | 0.7 × 交叉熵 + 0.3 × 监督对比损失（τ = 0.3），键来自动量编码器（含投影头）和队列（4,096 个）；同一样本不当自己的正例，没有正例的样本跳过对比项 |
#
# 共同的配方（和第二组 `e12` 相同）：`microsoft/mdeberta-v3-base`，`[CLS] text1 [SEP] text2 [SEP]`，最长 96 个 token，lr 2e-5，batch 32，前 10% 步数升温后线性衰减，weight decay 0.01（偏置和 LayerNorm 不加），梯度裁剪 1.0，最多 12 轮，每轮在内层验证上按原方向 WF1 选轮（同分取早的），推理用 fp32、相同输入只算一次。数据划分和第二组、第四组相同：按英语训练集的家族分 5 折（种子 2027），每折里再按家族分出 10% 作内层验证；四个赛道用同样的 pair_id。
#
# **和第二组 notebook（`dico_diagnostics`）的区别**：
#
# 1. 混合精度：支持 bf16 的卡（A10、4090）用 bf16、不用梯度缩放；T4 仍用 fp16。精度、卡、平台、torch 版本记在 `run_meta.json` 和每次运行的 `done.json` 里。
# 2. 每次运行（赛道 × 方法 × 种子 × 折）是一个独立任务，有自己的 `task.json`（全部训练变量），跑完写 `done.json`；被打断的任务从头重跑（一次只要几分钟）。
# 3. 模型快照：第一次运行时记下 Hugging Face 上 `main` 的 commit，之后固定用它。在 PAI-DSW 上先试 ModelScope，下载的文件和这个快照逐个核对，对不上或下载失败就改用 hf-mirror。
# 4. fold 0 的模型同时评官方 dev（训练 2,190 行），可以和同样训练数据量的 Qwen 比较。
#
# **三个阶段**（`STAGE`，PAI-DSW 上由 sh 脚本传入）：
#
# - `smoke`：每种方法在小子集上跑 2 轮，模拟一次中断再续跑，用真实的 torch 核对各个损失的写法（和逐条循环的参考实现比），再用真实规模的数据测每种方法每步的秒数，打印第一、第二阶段的时间估计。约 10 分钟（另加下载模型）。
# - `stage1`：英语 fold 0、种子 42–44：C0、B2、C1、C2、C3、C4 的两个 m；再加西班牙语、巴斯克语、跨语言的 C0（fold 0，3 个种子）。
# - `stage2`：英语 5 折、种子 42–46：C0、C1、C2、C3 和选定 m 的 C4（第一阶段已完成的会跳过）。每个种子把 5 折拼成 3,042 行，再用官方脚本算一次分数。
#
# **C4 的 m 怎么选（事先定好）**：第一阶段两个 m 各跑 3 个种子，比较选中那一轮的内层验证原方向 WF1 的三种子平均（不看 fold 0、dev 和老师），大的胜出，相同取 0.99。第二阶段开始时自动按这个规则选，并写进 `moco_m_choice.json`。
#
# **时间**：SMOKE 会按实测打印。第二组在 T4 上每个 12 轮的英语运行约 3.1 分钟（每轮 14.7 秒）。

# %%
import os
import sys
from pathlib import Path

if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
    raise RuntimeError("CUDA 已初始化。请重启会话，再从头运行。")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

STAGE = os.environ.get("DICO_STAGE", "smoke")  # smoke / stage1 / stage2；notebook 里直接改这里
MODEL = "microsoft/mdeberta-v3-base"
SEEDS_STAGE1 = [42, 43, 44]
SEEDS_STAGE2 = [42, 43, 44, 45, 46]

# 共同的训练配方（和第二组 e12 相同）
MAX_EPOCHS = 12
LR = 2e-5
BATCH_SIZE = 32
EVAL_BATCH = 64
MAX_LEN = 96
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
FOLDS, FOLD_SEED, INNER_FRACTION = 5, 2027, 0.1

# 各方法的超参数（事先固定，不在结果出来后调）
KD = {"alpha": 0.5, "T": 2.0}  # alpha 是 KL 项的权重
RDROP = {"alpha": 1.0}
DIR_RDROP = {"alpha": 1.0, "labels": ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"]}
PROJ = {"hidden": 768, "out": 128}  # [CLS] → Linear → ReLU → Linear → L2 归一化
SCL = {"lambda": 0.3, "tau": 0.3}
MOCO = {"lambda": 0.3, "tau": 0.3, "queue": 4096, "m_candidates": [0.99, 0.999], "m_tie": 0.99}
TEACHER_SOURCE_FINGERPRINT = "fcb2485b743956822b75c3703943ec806bc22ba955897c87082d0e6b8608566c"  # 第四组英语 3 轮

SMOKE_ROWS = {"fit": 96, "val": 32, "test": 48, "dev": 48}
SMOKE_EPOCHS = 2
TIMING_STEPS = 12  # SMOKE 测速：每种方法用真实规模的 batch 跑多少步（前 3 步不计）

ON_KAGGLE = Path("/kaggle/working").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = not ON_KAGGLE and Path("/var/colab/hostname").exists()
ON_PAI = not ON_KAGGLE and not ON_COLAB and Path("/mnt/workspace").is_dir()
if "DICO_WORK" in os.environ:
    WORK = os.path.abspath(os.environ["DICO_WORK"])
elif ON_PAI:
    WORK = "/mnt/workspace/dico_mdeberta"
elif ON_KAGGLE:
    WORK = "/kaggle/working"
else:
    WORK = os.path.abspath(".")
PLATFORM = os.environ.get("DICO_PLATFORM") or ("Kaggle" if ON_KAGGLE else "Colab" if ON_COLAB else "阿里云 PAI-DSW" if ON_PAI else "其他 Linux")
os.makedirs(WORK, exist_ok=True)
if ON_PAI:
    os.environ.setdefault("HF_HOME", os.path.join(WORK, "hf_cache"))
if not ON_KAGGLE and not ON_COLAB and "HF_ENDPOINT" not in os.environ:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/" + MODEL, timeout=10)
    except Exception as e:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print(f"连不上 huggingface.co（{type(e).__name__}），改用镜像 https://hf-mirror.com")
DICO_DIR = os.environ.get("DICO_DIR", os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI"))
TEACHER_DIR = os.environ.get("DICO_TEACHER", os.path.join(WORK, "dico_qwen_teacher_results"))
OUT = os.path.join(WORK, "dico_mdeberta_smoke" if STAGE == "smoke" else "dico_mdeberta_results")
print("运行平台：", PLATFORM, "；阶段：", STAGE, "；结果目录：", OUT)

# %%
import csv
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import shutil
import subprocess
import time
import zipfile
from collections import Counter

VERSION = "dico_mdeberta_methods-1"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
SWAP = [0, 2, 1, 3]  # 对调两句后：FORWARD ↔ BACKWARD，其余不变
METRICS = ["weighted_f1", "soft_cons", "hard_cons"]
REFERENCE_FIELDS = ["instance_id", "pair_id", "reverse_pair_id", "label"]  # 官方评分脚本用 csv.Sniffer 猜格式，只写这四列
RUN_STAMP = time.strftime("%Y%m%d-%H%M")
STAGE1_TRACK1 = ["ce", "kd", "rdrop", "dir_rdrop", "scl", "moco_m0.99", "moco_m0.999"]
assert STAGE in ("smoke", "stage1", "stage2"), STAGE
SMOKE = STAGE == "smoke"


# ---------- 文件、数据 ----------

def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def pip_install(*pkgs):
    index = ["-i", "http://mirrors.cloud.aliyuncs.com/pypi/simple/", "--trusted-host", "mirrors.cloud.aliyuncs.com"] if ON_PAI else []
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *index, *pkgs], check=True)


def ensure_packages():
    missing = [p for p, m in (("sentencepiece", "sentencepiece"), ("protobuf", "google.protobuf")) if importlib.util.find_spec(m.split(".")[0]) is None]
    if missing:
        pip_install(*missing)


def ensure_official_repo():
    """先找已解压的官方仓库；没有就找 WORK 里官方仓库的 zip（PAI-DSW 上连 GitHub 很慢，由上传包带来），最后才克隆。"""
    if os.path.isdir(os.path.join(DICO_DIR, "final_data")):
        return
    zips = sorted(glob.glob(os.path.join(WORK, "SemEval-2027-Task-2-DiCo-NLI*.zip")))
    if zips:
        tmp = DICO_DIR + ".unzip"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zips[-1]) as z:
            z.extractall(tmp)
        roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(tmp, "**", "final_data", "train"), recursive=True)]
        if len(roots) != 1:
            raise SystemExit(f"{zips[-1]} 里找不到唯一的 final_data 目录")
        shutil.move(roots[0], DICO_DIR)
        shutil.rmtree(tmp, ignore_errors=True)
        print("已从 zip 解压官方仓库：", zips[-1])
        return
    try:
        res = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git", DICO_DIR],
                             capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        res = subprocess.CompletedProcess([], 1, "", "git clone 超过 10 分钟没完成")
    if res.returncode != 0:
        shutil.rmtree(DICO_DIR, ignore_errors=True)
        raise SystemExit(f"克隆官方仓库失败：{(res.stdout + res.stderr).strip()[-300:]}\n"
                         f"请把官方仓库的 zip（GitHub 上 Code → Download ZIP）放进 {WORK}，再重新运行。")


def data_file(track, split, kind):
    return os.path.join(DICO_DIR, "final_data", split, f"dico_nli_{split}_track{track}_{kind}.csv")


def data_hashes(track):
    return {os.path.relpath(data_file(track, s, k), DICO_DIR): sha256(data_file(track, s, k))
            for s in ("train", "dev") for k in ("participant_labeled", "reference")}


# ---------- 家族、折、划分（和第二组、第四组逐行相同） ----------

def norm(s):
    return " ".join(s.lower().split())


def build_families(rows):
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


_SPLIT = {}


def split_base():
    if not _SPLIT:
        train1 = read_rows(data_file(1, "train", "participant_labeled"))
        fam_of = build_families(train1)
        _SPLIT.update(fam_of=fam_of, fold_of=assign_folds(train1, fam_of))
    return _SPLIT["fam_of"], _SPLIT["fold_of"]


def smoke_pairs(rows, n):
    """按 pair 整对抽样（反向行都在子集里，官方评分脚本才能算一致性），先保证四类各一对。"""
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


def parts_for(track, fold, with_dev, smoke=SMOKE):
    """fit / val / test（被预测的那一折）/ dev。内层验证的种子和第二组相同：FOLD_SEED * 100 + fold + 1。"""
    fam_of, fold_of = split_base()
    train = read_rows(data_file(track, "train", "reference"))  # reference 比 participant_labeled 多一列 reverse_pair_id，其余相同
    rest = {fam_of[r["pair_id"]] for r in train if fold_of[fam_of[r["pair_id"]]] != fold}
    inner = pick_families(rest, INNER_FRACTION, FOLD_SEED * 100 + fold + 1)
    parts = {"fit": [], "val": [], "test": []}
    for r in train:
        f = fam_of[r["pair_id"]]
        parts["test" if fold_of[f] == fold else "val" if f in inner else "fit"].append(r)
    if with_dev:
        parts["dev"] = read_rows(data_file(track, "dev", "reference"))
    if smoke:
        parts = {k: smoke_pairs(v, SMOKE_ROWS[k]) for k, v in parts.items()}
    return parts


def needed_inputs(rows, both=True):
    need, seen = [], set()
    for r in rows:
        for x in (((r["text1"], r["text2"]), (r["text2"], r["text1"])) if both else ((r["text1"], r["text2"]),)):
            if x not in seen:
                seen.add(x)
                need.append(x)
    return need


# ---------- 老师概率（B2） ----------

def load_teacher(seed, fit_rows):
    """第四组英语 3 轮 Qwen 的同种子老师在 fit 行上的四类概率；返回 (每行的概率, teacher.jsonl 的 SHA256)。"""
    fp_path = os.path.join(TEACHER_DIR, "fingerprint.json")
    path = os.path.join(TEACHER_DIR, "track1", f"seed{seed}", "teacher.jsonl")
    if not (os.path.exists(fp_path) and os.path.exists(path) and os.path.exists(os.path.join(os.path.dirname(path), "done.json"))):
        raise FileNotFoundError(f"找不到老师概率：{path}")
    fp = read_json(fp_path)
    if fp.get("source_fingerprint_sha256") != TEACHER_SOURCE_FINGERPRINT or fp.get("smoke"):
        raise SystemExit(f"{fp_path} 不是第四组英语 3 轮那次正式运行导出的老师")
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["part"] == "fit":
                table[(d["a"], d["b"])] = d["p"]
    probs = []
    for r in fit_rows:
        p = table.get((r["text1"], r["text2"]))
        if p is None:
            raise SystemExit(f"老师概率里没有这一行：{r['instance_id']}")
        assert len(p) == 4 and abs(sum(p) - 1) < 1e-4 and min(p) >= 0
        probs.append(p)
    return probs, sha256(path)


# ---------- 方法定义、任务 ----------

def method_params(method):
    if method == "ce":
        return {"loss": "ce"}
    if method == "kd":
        return {"loss": "0.5*CE + 0.5*T^2*KL(teacher_T || student_T)", **KD,
                "teacher": "第四组英语 3 轮 Qwen SFT，同种子，fit 行上的四类概率", "teacher_source_fingerprint": TEACHER_SOURCE_FINGERPRINT}
    if method == "rdrop":
        return {"loss": "(CE1+CE2)/2 + alpha*symKL(p1,p2)", **RDROP}
    if method == "dir_rdrop":
        return {"loss": "CE(A,B) + alpha*symKL(p(A,B), swap(p(B,A)))，只对金标在 labels 里的行", **DIR_RDROP}
    if method == "scl":
        return {"loss": "(1-lambda)*CE + lambda*SupCon(batch)", **SCL, "proj": PROJ, "rep": "最后一层 [CLS]"}
    if method.startswith("moco_m"):
        m = float(method[len("moco_m"):])
        return {"loss": "(1-lambda)*CE + lambda*SupCon(q, 动量编码器的键 + 队列)", "lambda": MOCO["lambda"], "tau": MOCO["tau"],
                "queue": MOCO["queue"], "m": m, "proj": PROJ, "rep": "最后一层 [CLS]",
                "target": "动量编码器（含投影头）eval 模式，不回传梯度；同一样本不当自己的正例"}
    raise ValueError(method)


def task_id(t):
    return f"track{t['track']}/{t['method']}/seed{t['seed']}/fold{t['fold']}"


def task_dir(t):
    return os.path.join(OUT, task_id(t))


def make_task(track, method, seed, fold, with_dev):
    return {"track": track, "method": method, "seed": seed, "fold": fold, "with_dev": with_dev}


def choose_m():
    """C4 的 m：第一阶段两个 m 各 3 个种子，选中那一轮的内层验证 WF1 的平均，大的胜出，相同取 MOCO['m_tie']。"""
    path = os.path.join(OUT, "moco_m_choice.json")
    if os.path.exists(path):
        return read_json(path)["m"]
    means = {}
    for m in MOCO["m_candidates"]:
        vals = []
        for s in SEEDS_STAGE1:
            d = os.path.join(OUT, f"track1/moco_m{m}/seed{s}/fold0/done.json")
            if not os.path.exists(d):
                raise SystemExit(f"第二阶段要先选 m，但第一阶段的 moco_m{m} seed{s} 还没跑完。请先跑完第一阶段。")
            vals.append(read_json(d)["val_weighted_f1"])
        means[m] = sum(vals) / len(vals)
    best = max(means.values())
    winners = [m for m in MOCO["m_candidates"] if abs(means[m] - best) < 1e-12]
    m = MOCO["m_tie"] if len(winners) > 1 else winners[0]
    write_json(path, {"m": m, "inner_val_wf1_mean": means, "rule": choose_m.__doc__})
    print("C4 的 m 按事先定的规则选为", m, "；内层验证 WF1 的三种子平均：", means)
    return m


def stage_tasks(stage):
    if stage == "smoke":
        return ([make_task(1, m, 42, 0, True) for m in STAGE1_TRACK1]
                + [make_task(2, "ce", 42, 0, True), make_task(4, "ce", 42, 0, True)])
    if stage == "stage1":
        tasks = [make_task(1, m, s, 0, True) for s in SEEDS_STAGE1 for m in STAGE1_TRACK1]
        tasks += [make_task(t, "ce", s, 0, True) for t in (2, 3, 4) for s in SEEDS_STAGE1]
        return tasks
    m = choose_m()
    methods = ["ce", "rdrop", "dir_rdrop", "scl", f"moco_m{m}"]
    return [make_task(1, meth, s, k, k == 0) for s in SEEDS_STAGE2 for k in range(FOLDS) for meth in methods]


# ---------- 指纹、运行记录、结果包 ----------

def common_fingerprint():
    return {"version_family": "dico_mdeberta_methods", "model": MODEL,
            "train": {"max_epochs": MAX_EPOCHS, "lr": LR, "batch": BATCH_SIZE, "eval_batch": EVAL_BATCH, "max_len": MAX_LEN,
                      "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                      "selection": "内层验证原方向 WF1，同分取早的"},
            "split": {"folds": FOLDS, "fold_seed": FOLD_SEED, "inner_fraction": INNER_FRACTION, "inner_seed": "FOLD_SEED*100+fold+1"},
            "smoke": SMOKE}


def task_fingerprint(t, revision, extra=None):
    fp = {**common_fingerprint(), "model_revision": revision, "track": t["track"], "fold": t["fold"], "seed": t["seed"],
          "method": t["method"], "method_params": method_params(t["method"]), "with_dev": t["with_dev"],
          "data_sha256": data_hashes(t["track"]),
          "epochs": SMOKE_EPOCHS if SMOKE else MAX_EPOCHS, "smoke_rows": SMOKE_ROWS if SMOKE else None}
    fp.update(extra or {})
    return fp


def restore_previous_results():
    """第二阶段在新实例上跑时，从上传包里带来的第一阶段结果包恢复（只认设置相同、完成最多的那个）。"""
    if SMOKE or os.path.isdir(OUT):
        return
    base = os.path.basename(OUT)
    found = sorted(set(glob.glob(os.path.join(WORK, f"{base}_*.zip")) + glob.glob(f"/kaggle/input/**/{base}_*.zip", recursive=True)))
    best, best_n = None, -1
    for path in found:
        try:
            with zipfile.ZipFile(path) as z:
                old = json.loads(z.read(f"{base}/fingerprint.json"))
                if {k: v for k, v in old.items() if k != "model_revision"} != common_fingerprint():
                    print("  跳过设置不同的结果包：", path)
                    continue
                n = sum(x.endswith("done.json") for x in z.namelist())
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            print("  跳过读不了的结果包：", path)
            continue
        if n > best_n:
            best, best_n = path, n
    if best:
        zipfile.ZipFile(best).extractall(WORK)
        print(f"已从结果包恢复（已完成 {best_n} 项）：", best)


def check_common_fingerprint():
    path = os.path.join(OUT, "fingerprint.json")
    fp = common_fingerprint()
    if os.path.exists(path):
        old = read_json(path)
        if {k: v for k, v in old.items() if k != "model_revision"} != fp:
            if SMOKE:
                shutil.rmtree(OUT)
            else:
                raise SystemExit(f"{path} 的设置和现在的不同。要接着跑，请把设置改回去；要重新开始，请删除或改名 {OUT}。")
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(path):
        write_json(path, fp)
    return read_json(path)


def code_sha256():
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
    """同一个结果目录只用同一份代码；SMOKE 的目录代码变了就清空重来。"""
    path = os.path.join(OUT, "run_meta.json")
    if code_hash is None:
        if not SMOKE:
            raise SystemExit("取不到这次运行的代码哈希，正式运行先停下。请把输出发给 Claude。")
        return
    old = {r.get("code_sha256") for r in (read_json(path) if os.path.exists(path) else [])} - {None}
    if not old or old == {code_hash} or os.environ.get("DICO_ALLOW_NEW_CODE") == code_hash:
        return
    if SMOKE:
        fp = read_json(os.path.join(OUT, "fingerprint.json"))
        shutil.rmtree(OUT)
        write_json(os.path.join(OUT, "fingerprint.json"), fp)
        return
    raise SystemExit(f"这个结果目录是用另一份代码（{sorted(old)}）跑的，现在的代码哈希是 {code_hash}。不把新旧代码的输出混在一起。"
                     f"要重新开始，请删除或改名 {OUT}。")


def env_info():
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for name in ("transformers", "sentencepiece", "huggingface_hub", "modelscope", "numpy"):
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = None
    return info


def record_run(info):
    path = os.path.join(OUT, "run_meta.json")
    code_hash, source = code_sha256()
    check_code(code_hash)
    runs = read_json(path) if os.path.exists(path) else []
    runs.append({"start": RUN_STAMP, "version": VERSION, "stage": STAGE, "code_sha256": code_hash, "code_source": source,
                 "platform": PLATFORM, "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"), **info})
    write_json(path, runs)
    print("代码哈希：", code_hash, f"（{source}）")
    return code_hash


_LAST_PACK = [0.0]


def pack(force=True):
    """结果包：整个结果目录（只有 logits、预测、分数和记录，没有模型权重），加上 sh 的日志。"""
    if not force and time.time() - _LAST_PACK[0] < 600:  # 跑的过程中最多每 10 分钟更新一次
        return None
    _LAST_PACK[0] = time.time()
    base = os.path.basename(OUT)
    zip_path = os.path.join(WORK, f"{base}_{RUN_STAMP}.zip")
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)):
            if os.path.isfile(path) and not path.endswith(".tmp"):
                z.write(path, os.path.relpath(path, WORK))
        log = os.environ.get("DICO_LOG")
        if log and os.path.exists(log):
            z.write(log, f"{base}/logs/{os.path.basename(log)}")
    os.replace(tmp, zip_path)
    return zip_path


# ---------- 模型下载（固定快照） ----------

WEIGHT_PATTERNS = ["*.json", "*.model", "*.safetensors", "pytorch_model.bin"]


def wanted(name):
    import fnmatch
    return "/" not in name and any(fnmatch.fnmatch(name, p) for p in WEIGHT_PATTERNS)


def file_digest(path, algo, prefix=b""):
    h = hashlib.new(algo, prefix)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch_model(revision):
    """返回 (本地模型目录, Hugging Face 快照的 commit, 来源)。revision 为 None 时取 main 当前的 commit 并固定下来。"""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL, revision=revision or "main", files_metadata=True)
    files = [s for s in info.siblings if wanted(s.rfilename)]
    if ON_PAI:
        try:
            from modelscope import snapshot_download as ms_download
            try:
                path = ms_download(MODEL, cache_dir=os.path.join(WORK, "ms_cache"), allow_patterns=WEIGHT_PATTERNS)
            except TypeError:  # 旧版 modelscope 的参数名
                path = ms_download(MODEL, cache_dir=os.path.join(WORK, "ms_cache"), allow_file_pattern=WEIGHT_PATTERNS)
            bad = []
            for s in files:
                f = os.path.join(path, s.rfilename)
                if not os.path.isfile(f):
                    bad.append(f"{s.rfilename}：ModelScope 上没有")
                    continue
                lfs = s.lfs
                want = (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256) if lfs else s.blob_id
                got = file_digest(f, "sha256") if lfs else file_digest(f, "sha1", b"blob %d\0" % os.path.getsize(f))
                if got != want:
                    bad.append(f"{s.rfilename}：内容不同")
            if bad or not files:
                raise RuntimeError(f"和 Hugging Face 快照 {info.sha} 对不上：{bad[:5]}")
            print(f"从 ModelScope 下载，{len(files)} 个文件和 Hugging Face 快照 {info.sha} 逐个核对一致", flush=True)
            return path, info.sha, "modelscope（已按 Hugging Face 快照核对）"
        except Exception as e:
            print(f"从 ModelScope 取模型没成功（{type(e).__name__}: {str(e)[:200]}），改从 {os.environ.get('HF_ENDPOINT', 'Hugging Face')} 下载", flush=True)
    path = snapshot_download(MODEL, revision=info.sha, allow_patterns=WEIGHT_PATTERNS)
    return path, info.sha, os.environ.get("HF_ENDPOINT", "https://huggingface.co")


# ---------- 概率、指标、评分 ----------

def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]


def probs(lm, a, b, mode):
    pa = softmax(lm[(a, b)])
    if mode == "plain":
        return pa
    pb = softmax(lm[(b, a)])
    return [(pa[i] + pb[SWAP[i]]) / 2 for i in range(4)]


def argmax_label(p):
    return LABELS[max(range(4), key=lambda i: p[i])]


def weighted_f1(gold, pred):
    total, out = len(gold), 0.0
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        out += (2 * tp / (2 * tp + fp + fn) if tp else 0.0) * sum(g == l for g in gold) / total
    return out


def official_score(rows, labels, d, name):
    """官方脚本评分。金标子集只写四列（短语以撇号开头时 csv.Sniffer 会猜错格式）。"""
    part = name.rsplit("_", 1)[0]  # test_plain 和 test_sym 共用一份金标
    ref, pred, out = (os.path.abspath(os.path.join(d, x)) for x in (f"reference_{part}.csv", f"pred_{name}.csv", f"scores_{name}"))
    if not os.path.exists(ref):
        write_rows(ref, rows, REFERENCE_FIELDS)
    write_rows(pred, [{"instance_id": r["instance_id"], "label": l} for r, l in zip(rows, labels)], ["instance_id", "label"])
    res = subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", ref, "--predictions", pred, "--output-dir", out],
                         cwd=DICO_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"官方评分脚本出错（{name}）：{res.stderr.strip()[-500:]}")
    s = read_json(os.path.join(out, "scores.json"))
    local = weighted_f1([r["label"] for r in rows], labels)
    assert abs(local - s["weighted_f1"]) < 1e-9, (name, local, s["weighted_f1"])
    return {k: s[k] for k in METRICS}


def save_logits(path, lm):
    write_json(path, {"inputs": [list(k) for k in lm], "logits": list(lm.values())})


def load_logits(path):
    d = read_json(path)
    return {tuple(k): v for k, v in zip(d["inputs"], d["logits"])}


# ---------- 损失（唯一和方法有关的部分） ----------

def sym_kl(lp, lq):
    """(KL(p‖q) + KL(q‖p)) / 2，按行平均；输入是 log 概率（fp32）。"""
    p, q = lp.exp(), lq.exp()
    return 0.5 * ((p * (lp - lq)).sum(-1) + (q * (lq - lp)).sum(-1)).mean()


def supcon(q, q_labels, q_ids, keys, k_labels, k_ids, tau):
    """监督对比损失。锚点 q 的候选是 keys 里样本编号不同的键，正例是其中标签相同的；
    L_i = -mean_{p∈P(i)} log( exp(q_i·k_p/τ) / Σ_{a∈A(i)} exp(q_i·k_a/τ) )，没有正例的锚点跳过。返回 (损失, 有效锚点数)。"""
    import torch
    logits = q @ keys.T / tau
    valid = k_ids[None, :] != q_ids[:, None]
    pos = valid & (k_labels[None, :] == q_labels[:, None])
    logits = logits.masked_fill(~valid, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    npos = pos.sum(1)
    has = npos > 0
    if not bool(has.any()):
        return q.sum() * 0.0, 0
    per = -(log_prob.masked_fill(~pos, 0.0).sum(1)) / npos.clamp_min(1)
    return per[has].mean(), int(has.sum())


def kd_loss(logits, y, teacher_p, alpha, T):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, y)
    t = torch.softmax(torch.log(teacher_p.clamp_min(1e-12)) / T, dim=-1)
    s_log = torch.log_softmax(logits / T, dim=-1)
    kl = (t * (torch.log(t.clamp_min(1e-12)) - s_log)).sum(-1).mean()
    return (1 - alpha) * ce + alpha * T * T * kl, {"ce": float(ce), "kl": float(kl)}


def self_check_losses():
    """SMOKE 用：在真实的 torch 上，把各个损失和逐条循环写的参考实现比一遍。"""
    import torch
    g = torch.Generator().manual_seed(0)
    n, d = 7, 5
    q = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)
    keys = torch.nn.functional.normalize(torch.randn(11, d, generator=g), dim=1)
    ql, qi = torch.tensor([0, 1, 1, 2, 3, 0, 2]), torch.arange(n)
    kl_, ki = torch.tensor([0, 1, 2, 3, 0, 1, 1, 2, 3, 0, 3]), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    got, _ = supcon(q, ql, qi, keys, kl_, ki, 0.3)
    tot, cnt = 0.0, 0
    for i in range(n):
        cand = [j for j in range(len(keys)) if int(ki[j]) != int(qi[i])]
        pos = [j for j in cand if int(kl_[j]) == int(ql[i])]
        if not pos:
            continue
        den = sum(math.exp(float(q[i] @ keys[j]) / 0.3) for j in cand)
        tot += -sum(math.log(math.exp(float(q[i] @ keys[j]) / 0.3) / den) for j in pos) / len(pos)
        cnt += 1
    assert abs(float(got) - tot / cnt) < 1e-4, ("supcon", float(got), tot / cnt)
    a, b = torch.randn(4, 4, generator=g), torch.randn(4, 4, generator=g)
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    ref = sum(0.5 * sum(math.exp(la[i, c]) * (la[i, c] - lb[i, c]) + math.exp(lb[i, c]) * (lb[i, c] - la[i, c]) for c in range(4))
              for i in range(4)) / 4
    assert abs(float(sym_kl(la, lb)) - float(ref)) < 1e-5, "sym_kl"
    tp = torch.softmax(torch.randn(4, 4, generator=g), -1)
    y = torch.tensor([0, 1, 2, 3])
    got, _ = kd_loss(a, y, tp, 0.5, 2.0)
    ref = 0.0
    for i in range(4):
        ce = -float(torch.log_softmax(a[i], -1)[y[i]])
        t = [math.exp(math.log(float(tp[i, c])) / 2) for c in range(4)]
        z = sum(t)
        t = [x / z for x in t]
        s = torch.log_softmax(a[i] / 2, -1)
        ref += 0.5 * ce + 0.5 * 4 * sum(t[c] * (math.log(t[c]) - float(s[c])) for c in range(4))
    assert abs(float(got) - ref / 4) < 1e-4, ("kd", float(got), ref / 4)
    pab = torch.randn(3, 4, generator=g)
    swapped = torch.log_softmax(pab, -1)[:, SWAP]
    assert torch.allclose(swapped[:, 1], torch.log_softmax(pab, -1)[:, 2])
    q2 = Queue(5, 3, "cpu")
    for step in range(4):
        q2.push(torch.ones(2, 3) * step, torch.tensor([step, step]), torch.tensor([10 * step, 10 * step + 1]), step)
    assert q2.filled == 5 and sorted(q2.ids[: q2.filled].tolist()) == [11, 20, 21, 30, 31], q2.ids.tolist()
    print("损失自检通过：监督对比、对称 KL、蒸馏、F/B 对调、队列的环形写入都和逐条参考实现一致", flush=True)


class Queue:
    """MoCo 的键队列：存键、标签、样本编号和入队时的步数，环形覆盖最旧的。"""

    def __init__(self, size, dim, device):
        import torch
        self.size, self.ptr, self.filled = size, 0, 0
        self.keys = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size, dtype=torch.long, device=device)
        self.ids = torch.full((size,), -1, dtype=torch.long, device=device)
        self.steps = torch.zeros(size, dtype=torch.long, device=device)

    def push(self, k, labels, ids, step):
        n = k.shape[0]
        idx = [(self.ptr + i) % self.size for i in range(n)]
        self.keys[idx] = k.detach().float()
        self.labels[idx] = labels
        self.ids[idx] = ids
        self.steps[idx] = step
        self.ptr = (self.ptr + n) % self.size
        self.filled = min(self.size, self.filled + n)

    def view(self):
        n = self.filled
        return self.keys[:n], self.labels[:n], self.ids[:n]

    def mean_age(self, step):
        return float((step - self.steps[: self.filled]).float().mean()) if self.filled else 0.0


# ---------- 训练 ----------

class SimulatedInterrupt(Exception):
    pass


def free_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


_MODEL_PATH = {}
_TOK = {}


def tokenizer():
    if "tok" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["tok"] = AutoTokenizer.from_pretrained(_MODEL_PATH["path"])
    return _TOK["tok"]


def amp_dtype():
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def train_run(t, parts, teacher=None, interrupt_after_epoch=None, max_steps=None):
    """训练一次，按内层验证选轮，返回 val / test / dev 的 logits 和训练记录。max_steps 只给 SMOKE 测速用。"""
    import copy
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    method, seed = t["method"], t["seed"]
    mp = method_params(method)
    epochs = SMOKE_EPOCHS if SMOKE else MAX_EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = tokenizer()
    fit, val = parts["fit"], parts["val"]
    if teacher is not None:
        assert len(teacher) == len(fit)

    def enc_of(t1, t2):
        e = tok(t1, t2, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in e.items()}

    @torch.no_grad()
    def logits_for(model, pairs):
        model.eval()
        lens = [len(x) for x in tok([a for a, _ in pairs], [b for _, b in pairs], truncation=True, max_length=MAX_LEN)["input_ids"]]
        order = sorted(range(len(pairs)), key=lambda i: (lens[i], i))
        out = [None] * len(pairs)
        for s in range(0, len(order), EVAL_BATCH):
            idx = order[s: s + EVAL_BATCH]
            e = enc_of([pairs[i][0] for i in idx], [pairs[i][1] for i in idx])
            for i, row in zip(idx, model(**e).logits.float().cpu().tolist()):
                out[i] = row
        return dict(zip(pairs, out))

    val_inputs = needed_inputs(val, both=False)

    def run(dtype):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_PATH["path"], num_labels=4).float().to(device)
        hidden = model.config.hidden_size
        proj = target = target_proj = queue = None
        if method == "scl" or method.startswith("moco_m"):  # 在分类头之后建，分类头的初始化和其他方法相同
            proj = torch.nn.Sequential(torch.nn.Linear(hidden, PROJ["hidden"]), torch.nn.ReLU(),
                                       torch.nn.Linear(PROJ["hidden"], PROJ["out"])).to(device)
        if method.startswith("moco_m"):
            target, target_proj = copy.deepcopy(model).eval(), copy.deepcopy(proj).eval()
            for p in list(target.parameters()) + list(target_proj.parameters()):
                p.requires_grad_(False)
            queue = Queue(mp["queue"], PROJ["out"], device)
        named = list(model.named_parameters()) + ([(f"proj.{n}", p) for n, p in proj.named_parameters()] if proj is not None else [])
        no_decay = ("bias", "LayerNorm.weight", "layernorm.weight")
        groups = [{"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": WEIGHT_DECAY},
                  {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=LR)
        loader = DataLoader(list(range(len(fit))), batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: b,
                            generator=torch.Generator().manual_seed(seed))
        total = epochs * len(loader)
        sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * total), total)
        use_scaler = dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        dir_set = {L2I[l] for l in DIR_RDROP["labels"]}
        best, best_epoch, best_state, history, step = -1.0, 0, None, [], 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        step_times = []
        for epoch in range(epochs):
            model.train()
            if proj is not None:
                proj.train()
            t0, stats = time.time(), {"loss": [], "extra": []}
            for idx in loader:
                rows = [fit[i] for i in idx]
                y = torch.tensor([L2I[r["label"]] for r in rows], device=device)
                t1, t2 = [r["text1"] for r in rows], [r["text2"] for r in rows]
                ts = time.time()
                with torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                    if method == "ce":
                        loss = F.cross_entropy(model(**enc_of(t1, t2)).logits.float(), y)
                        extra = {}
                    elif method == "kd":
                        tp = torch.tensor([teacher[i] for i in idx], device=device, dtype=torch.float32)
                        loss, extra = kd_loss(model(**enc_of(t1, t2)).logits.float(), y, tp, mp["alpha"], mp["T"])
                    elif method == "rdrop":
                        lg = model(**enc_of(t1 + t1, t2 + t2)).logits.float()
                        l1, l2 = lg[: len(rows)], lg[len(rows):]
                        ce = (F.cross_entropy(l1, y) + F.cross_entropy(l2, y)) / 2
                        kl = sym_kl(torch.log_softmax(l1, -1), torch.log_softmax(l2, -1))
                        loss, extra = ce + mp["alpha"] * kl, {"ce": float(ce), "kl": float(kl)}
                    elif method == "dir_rdrop":
                        sel = [i for i, r in enumerate(rows) if L2I[r["label"]] in dir_set]
                        lg = model(**enc_of(t1 + [t2[i] for i in sel], t2 + [t1[i] for i in sel])).logits.float()
                        lab = lg[: len(rows)]
                        ce = F.cross_entropy(lab, y)
                        if sel:
                            kl = sym_kl(torch.log_softmax(lab[sel], -1), torch.log_softmax(lg[len(rows):], -1)[:, SWAP])
                        else:
                            kl = lab.sum() * 0.0
                        loss, extra = ce + mp["alpha"] * kl, {"ce": float(ce), "kl": float(kl), "sel_frac": len(sel) / len(rows)}
                    else:  # scl / moco：分类和对比都用同一次前向
                        out = model(**enc_of(t1, t2), output_hidden_states=True)
                        ce = F.cross_entropy(out.logits.float(), y)
                        q = F.normalize(proj(out.hidden_states[-1][:, 0]).float(), dim=1)
                        ids = torch.tensor(idx, device=device)
                        if method == "scl":
                            con, n_anchor = supcon(q, y, ids, q, y, ids, mp["tau"])
                        else:
                            with torch.no_grad():
                                kout = target(**enc_of(t1, t2), output_hidden_states=True)
                                k = F.normalize(target_proj(kout.hidden_states[-1][:, 0]).float(), dim=1)
                            qk, ql, qi = queue.view()
                            con, n_anchor = supcon(q, y, ids, torch.cat([k, qk]), torch.cat([y, ql]), torch.cat([ids, qi]), mp["tau"])
                        loss = (1 - mp["lambda"]) * ce + mp["lambda"] * con
                        extra = {"ce": float(ce), "con": float(con), "anchors": n_anchor / len(rows)}
                if not torch.isfinite(loss):
                    return None  # 半精度出现非有限损失：整次改用 fp32 重跑
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if queue is not None:
                    with torch.no_grad():
                        m = mp["m"]
                        for pt, po in zip(list(target.parameters()) + list(target_proj.parameters()),
                                          list(model.parameters()) + list(proj.parameters())):
                            pt.mul_(m).add_(po.detach(), alpha=1 - m)
                    queue.push(k, y, ids, step)
                if device == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.time() - ts)
                stats["loss"].append(float(loss))
                stats["extra"].append(extra)
                if max_steps and len(step_times) >= max_steps:
                    return {"step_times": step_times}
            lm = logits_for(model, val_inputs)
            f1 = weighted_f1([r["label"] for r in val], [argmax_label(softmax(lm[(r["text1"], r["text2"])])) for r in val])
            rec = {"epoch": epoch + 1, "train_loss": sum(stats["loss"]) / len(stats["loss"]), "val_weighted_f1": f1,
                   "seconds": round(time.time() - t0, 1)}
            for key in sorted({k for e in stats["extra"] for k in e}):
                vals = [e[key] for e in stats["extra"] if key in e]
                rec[key] = sum(vals) / len(vals)
            if queue is not None:
                rec["queue_filled"] = queue.filled
                rec["queue_mean_key_age_steps"] = queue.mean_age(step)
            history.append(rec)
            print(f"    第 {epoch + 1} 轮：loss {rec['train_loss']:.4f}  内层验证 WF1 {f1:.4f}  {rec['seconds']} 秒", flush=True)
            if f1 > best:
                best, best_epoch = f1, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if interrupt_after_epoch == epoch + 1:
                raise SimulatedInterrupt()
        model.load_state_dict(best_state)
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None
        result = {"history": history, "best_epoch": best_epoch, "val_weighted_f1": best, "peak_memory_gib": peak,
                  "seconds_per_step": round(sum(step_times) / max(1, len(step_times)), 4), "steps": len(step_times),
                  "steps_per_epoch": len(loader)}
        result["val_logits"] = logits_for(model, val_inputs)
        result["test_logits"] = logits_for(model, needed_inputs(parts["test"]))
        if "dev" in parts:
            result["dev_logits"] = logits_for(model, needed_inputs(parts["dev"]))
        del model, best_state, target, opt
        free_memory()
        return result

    dtype = amp_dtype()
    res = run(dtype)
    precision = {None: "fp32", torch.bfloat16: "bf16 autocast", torch.float16: "fp16 autocast"}[dtype]
    if max_steps:
        return res
    if res is None:
        print("    出现非有限损失，整次改用 fp32 重跑", flush=True)
        res, precision = run(None), "fp32（半精度出现非有限损失后重跑）"
    res["precision"] = precision
    return res


TRAIN_FN = train_run  # 测试时可以换成假的训练函数


def run_task(t, revision, code_hash, interrupt_after_epoch=None):
    d = task_dir(t)
    parts = parts_for(t["track"], t["fold"], t["with_dev"])
    teacher, extra = None, {}
    if t["method"] == "kd":
        teacher, tsha = load_teacher(t["seed"], parts["fit"])
        extra["teacher_sha256"] = tsha
    fp = task_fingerprint(t, revision, extra)
    if os.path.exists(os.path.join(d, "done.json")):
        old = read_json(os.path.join(d, "task.json"))
        if old != fp:
            diff = sorted(k for k in set(fp) | set(old) if fp.get(k) != old.get(k))
            raise SystemExit(f"{task_id(t)} 已完成，但设置和现在不同：{diff}。不把两种设置的结果混在一起，请检查。")
        return "skip"
    if os.path.isdir(d):
        print(f"  {task_id(t)} 上次没跑完，从头重跑", flush=True)
        shutil.rmtree(d)
    write_json(os.path.join(d, "task.json"), fp)
    t0 = time.time()
    print(f"开始：{task_id(t)}（训练 {len(parts['fit'])} 行，内层验证 {len(parts['val'])} 行，"
          f"被预测的折 {len(parts['test'])} 行{'，dev ' + str(len(parts['dev'])) + ' 行' if 'dev' in parts else ''}）", flush=True)
    res = TRAIN_FN(t, parts, teacher=teacher, interrupt_after_epoch=interrupt_after_epoch)
    save_logits(os.path.join(d, "logits_val.json"), res["val_logits"])
    save_logits(os.path.join(d, "logits_test.json"), res["test_logits"])
    scores = {}
    for name in ("test", "dev"):
        if f"{name}_logits" not in res:
            continue
        if name == "dev":
            save_logits(os.path.join(d, "logits_dev.json"), res["dev_logits"])
        rows = parts[name]
        for mode in ("plain", "sym"):
            labels = [argmax_label(probs(res[f"{name}_logits"], r["text1"], r["text2"], mode)) for r in rows]
            scores.setdefault(name, {})[mode] = official_score(rows, labels, d, f"{name}_{mode}")
    write_json(os.path.join(d, "history.json"), res["history"])
    done = {"task": task_id(t), "best_epoch": res["best_epoch"], "val_weighted_f1": res["val_weighted_f1"], "scores": scores,
            "precision": res["precision"], "peak_memory_gib": res["peak_memory_gib"], "seconds_per_step": res["seconds_per_step"],
            "steps": res["steps"], "steps_per_epoch": res["steps_per_epoch"], "code_sha256": code_hash,
            "rows": {k: len(v) for k, v in parts.items()}, "minutes": round((time.time() - t0) / 60, 2)}
    write_json(os.path.join(d, "done.json"), done)
    s = scores["test"]["sym"]
    print(f"完成：{task_id(t)}  选中第 {res['best_epoch']} 轮  被预测的折 sym {s['weighted_f1']:.3f} / {s['soft_cons']:.3f} / {s['hard_cons']:.3f}"
          + (f"  dev sym {scores['dev']['sym']['weighted_f1']:.3f}" if "dev" in scores else "")
          + f"  用时 {done['minutes']} 分钟", flush=True)
    return "done"


# ---------- 汇总 ----------

def assemble_oof(track, method, seed):
    """5 折都完成时，每行取它所在折的模型的 logits，拼成整个训练集的折外预测，用官方脚本算一次。"""
    dirs = [os.path.join(OUT, f"track{track}/{method}/seed{seed}/fold{k}") for k in range(FOLDS)]
    if not all(os.path.exists(os.path.join(x, "done.json")) for x in dirs):
        return None
    out_dir = os.path.join(OUT, f"track{track}/{method}/seed{seed}/oof")
    if os.path.exists(os.path.join(out_dir, "done.json")):
        return read_json(os.path.join(out_dir, "done.json"))["scores"]
    fam_of, fold_of = split_base()
    rows = read_rows(data_file(track, "train", "reference"))
    lms = [load_logits(os.path.join(x, "logits_test.json")) for x in dirs]
    scores = {}
    for mode in ("plain", "sym"):
        labels = [argmax_label(probs(lms[fold_of[fam_of[r["pair_id"]]]], r["text1"], r["text2"], mode)) for r in rows]
        scores[mode] = official_score(rows, labels, out_dir, f"oof_{mode}")
    write_json(os.path.join(out_dir, "done.json"), {"scores": scores, "rows": len(rows)})
    return scores


def summarize():
    summary = {"version": VERSION, "stage": STAGE, "runs": {}, "oof": {}}
    for p in sorted(glob.glob(os.path.join(OUT, "track*/*/seed*/fold*/done.json"))):
        d = read_json(p)
        summary["runs"][d["task"]] = {k: d[k] for k in ("best_epoch", "val_weighted_f1", "scores", "precision", "minutes", "seconds_per_step")}
    for track_dir in sorted(glob.glob(os.path.join(OUT, "track*"))):
        track = int(os.path.basename(track_dir)[5:])
        for method_dir in sorted(glob.glob(os.path.join(track_dir, "*"))):
            for seed_dir in sorted(glob.glob(os.path.join(method_dir, "seed*"))):
                sc = assemble_oof(track, os.path.basename(method_dir), int(os.path.basename(seed_dir)[4:]))
                if sc:
                    summary["oof"][f"track{track}/{os.path.basename(method_dir)}/{os.path.basename(seed_dir)}"] = sc
    write_json(os.path.join(OUT, "summary.json"), summary)
    groups = {}
    for tid, r in summary["runs"].items():
        track, method, seed, fold = tid.split("/")
        if fold == "fold0":
            groups.setdefault((track, method), []).append(r)
    print("\n========== fold 0 汇总（对称化 WF1 / SoftCons / HardCons，种子平均；dev 只报告） ==========")
    for (track, method), rs in sorted(groups.items()):
        m = {k: sum(r["scores"]["test"]["sym"][k] for r in rs) / len(rs) for k in METRICS}
        dev = [r["scores"]["dev"]["sym"]["weighted_f1"] for r in rs if "dev" in r["scores"]]
        print(f"{track} {method:12s} {len(rs)} 个种子  fold 0 {m['weighted_f1']:.3f} / {m['soft_cons']:.3f} / {m['hard_cons']:.3f}"
              + (f"  dev WF1 {sum(dev) / len(dev):.3f}" if dev else ""))
    if summary["oof"]:
        print("5 折拼起来的折外预测（对称化 WF1）：")
        for k, sc in summary["oof"].items():
            print(f"  {k}: {sc['sym']['weighted_f1']:.4f}")
    return summary


# ---------- SMOKE：自检、中断续跑、测速、时间估计 ----------

def smoke_extras(revision, code_hash):
    import torch
    report = {}
    self_check_losses()
    report["loss_self_check"] = "通过"
    for s in SEEDS_STAGE1:  # 老师概率要覆盖真实的 2,190 行，不只是 SMOKE 子集
        probs_, tsha = load_teacher(s, parts_for(1, 0, False, smoke=False)["fit"])
        report[f"teacher_seed{s}"] = {"rows": len(probs_), "sha256": tsha}
    print("老师概率覆盖了三个种子的全部 2,190 行", flush=True)
    t = make_task(1, "ce", 43, 0, True)
    try:
        run_task(t, revision, code_hash, interrupt_after_epoch=1)
    except SimulatedInterrupt:
        free_memory()
    assert os.path.isdir(task_dir(t)) and not os.path.exists(os.path.join(task_dir(t), "done.json"))
    run_task(t, revision, code_hash)
    assert os.path.exists(os.path.join(task_dir(t), "done.json"))
    report["interrupt_resume"] = "通过：第 1 轮后模拟中断，重跑时从头训练并完成"
    print("中断续跑检查通过", flush=True)
    timing = {}
    probe = [(1, m) for m in STAGE1_TRACK1] + [(2, "ce"), (3, "ce"), (4, "ce")]
    for track, method in probe:
        full = parts_for(track, 0, False, smoke=False)  # 测速用正式规模的数据和 batch，只跑 TIMING_STEPS 步
        teacher = load_teacher(42, full["fit"])[0] if method == "kd" else None
        res = TRAIN_FN(make_task(track, method, 42, 0, False), full, teacher=teacher, max_steps=TIMING_STEPS)
        if res is None:
            raise SystemExit(f"测速时 track{track} {method} 出现非有限损失")
        per = sum(res["step_times"][3:]) / len(res["step_times"][3:])  # 前 3 步含初始化和缓存分配，不计
        timing[f"track{track}/{method}"] = {"seconds_per_step": round(per, 4), "steps": len(res["step_times"]) - 3}
        print(f"  测速 track{track} {method:12s} 每步 {per:.3f} 秒（{len(res['step_times']) - 3} 步，不计前 3 步）", flush=True)
        del res
        free_memory()
    report["timing"] = timing
    report["estimate"] = estimate(timing)
    write_json(os.path.join(OUT, "smoke_report.json"), report)
    free_memory()
    return report


def estimate(timing):
    """按测得的每步秒数估计正式运行的时间。每次运行另加约 0.3 分钟的载入模型和推理（按输入条数折算）。"""
    est = {}

    def run_min(track, method):
        n_fit = 2190 if track != 4 else 13140
        steps = math.ceil(n_fit / BATCH_SIZE) * MAX_EPOCHS
        infer = 0.3 * (1 if track != 4 else 6)
        return steps * timing[f"track{track}/{method}"]["seconds_per_step"] / 60 + infer

    s1 = sum(run_min(1, m) for m in STAGE1_TRACK1) * len(SEEDS_STAGE1) + sum(run_min(t, "ce") for t in (2, 3, 4)) * len(SEEDS_STAGE1)
    est["stage1_minutes"] = round(s1, 1)
    worst_m = max(MOCO["m_candidates"], key=lambda m: timing[f"track1/moco_m{m}"]["seconds_per_step"])
    per_seed_fold = sum(run_min(1, m) for m in ["ce", "rdrop", "dir_rdrop", "scl", f"moco_m{worst_m}"])
    n_new = len(SEEDS_STAGE2) * FOLDS - len(SEEDS_STAGE1)
    est["stage2_minutes"] = round(per_seed_fold * n_new, 1)
    est["per_run_minutes"] = {k: round(run_min(int(k[5]), k.split("/")[1]), 2) for k in timing}
    print(f"\n按实测估计：第一阶段约 {est['stage1_minutes']:.0f} 分钟，第二阶段约 {est['stage2_minutes']:.0f} 分钟（不含装环境、下载模型）", flush=True)
    return est


# ---------- 主流程 ----------

def main():
    ensure_packages()
    ensure_official_repo()
    restore_previous_results()
    common = check_common_fingerprint()
    revision = common.get("model_revision")
    path, revision_now, source = fetch_model(revision)
    if revision is None:
        common["model_revision"] = revision_now
        write_json(os.path.join(OUT, "fingerprint.json"), common)
    revision = revision_now
    _MODEL_PATH["path"] = path
    import torch
    dtype = amp_dtype()
    code_hash = record_run({**env_info(), "model_revision": revision, "model_source": source,
                            "amp": {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]})
    print("模型快照：", revision, "；来源：", source, "；混合精度：", dtype, flush=True)
    tasks = stage_tasks(STAGE)
    if any(t["method"] == "kd" for t in tasks):  # 开跑前先确认老师概率在，免得跑到一半才停
        for s in sorted({t["seed"] for t in tasks if t["method"] == "kd"}):
            try:
                load_teacher(s, parts_for(1, 0, False, smoke=False)["fit"])
            except FileNotFoundError as e:
                raise SystemExit(f"{e}。请把上传包里的 dico_qwen_teacher_results 放到 {TEACHER_DIR}（sh 的 setup 会自动放好）。")
    if SMOKE:
        smoke_extras(revision, code_hash)
    n_done = 0
    for i, t in enumerate(tasks, 1):
        status = run_task(t, revision, code_hash)
        n_done += status == "done"
        if status == "done":
            pack(force=False)
        print(f"[{i}/{len(tasks)}] {task_id(t)}：{'已完成，跳过' if status == 'skip' else '完成'}", flush=True)
    summarize()
    zip_path = pack()
    print("\n结果包：", zip_path)
    if SMOKE:
        print("SMOKE 跑完了：损失自检、中断续跑、各方法的小规模训练都通过。请把结果包和上面的输出发给 Claude，确认后再跑第一阶段。")
    elif STAGE == "stage1":
        print("第一阶段跑完了。请下载这个结果包发给 Claude；第二阶段在下一次开实例时跑（需要这个结果包）。")
    else:
        print("第二阶段跑完了。请下载这个结果包发给 Claude。")


if __name__ == "__main__":
    main()
