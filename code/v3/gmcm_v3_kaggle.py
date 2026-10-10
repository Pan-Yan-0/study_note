# %% [markdown]
# # GMCM 2026 E 题 · v3：DeBERTa-v3-large + 有序分布头 + 多种子集成（问题二、问题三），外加 v2 结论补验
#
# **这一版做什么**
# 1. **先补验 v2 的结论（只推理，不训练）**：90% 文本缺失档逐样本分析（按实际剩余词数分层、加入 joint 对照、按视频配对区间）；
#    原生缺失子集的门触发率、类别变化与 MAE 变化；逐种子、逐面板指标表。
# 2. **再冲性能**：文本主干换成 DeBERTa-v3-large（通用预训练模型，未用任何情感数据微调），全部线性层加 LoRA；
#    强度用 19 点有序分布头（网格 −3..3，步长 1/3，与标注格点一致），取分布中位数作为 MAE 最优点估计；
#    损失 = 加权交叉熵 + 有序分布交叉熵 + L1 + 软宏 F1 + R-Drop；训练时 30% 样本做连续段缺失增广；5 个种子取平均集成。
# 3. **事先锁定协议**：验证集按源视频对半分为“选择区 VS”（选 epoch、决策层、部署配置）与“校准区 VC”（只做共形校准）；
#    测试集只在所有选择完成后评估一次。协议在训练前写入 `PROTOCOL_v3.json`。
#
# **怎么跑**：Import Notebook；GPU T4、打开 Internet（首次下载 DeBERTa-v3-large 约 0.9 GB）；Add Input：比赛数据集 + 之前上传的检查点数据集
# （base/mg/joint 的 .pt 与 train_config.json；有 mg2 的 .pt 也一起放进去）。先 `SMOKE = True` 冒烟约 15 分钟，再 `SMOKE = False` 全量约 3 小时。
# 结果在 `/kaggle/working/outputs_v3.zip`；新训的检查点另打包为 `outputs_v3_ckpt.zip`，以后可以上传复用。

# %% [cfg] 0. 配置
SMOKE = False          # ← 第一次先改成 True 冒烟测试

import os, re, sys, json, math, time, glob, pickle, random, zipfile, hashlib, warnings, copy, shutil, subprocess
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 减少 DeBERTa 训练时的显存碎片

CFG = dict(
    SMOKE=SMOKE,
    INPUT_ROOT="/kaggle/input",
    OUT_DIR="/kaggle/working/outputs_v3",
    ATT2_PATH=None,               # 附件2 aligned_50.pkl（文件）；None 表示自动查找
    ATT3_PATH=None,               # 附件3 对齐版本（文件或文件夹）；None 表示自动查找
    ATT4_PATH=None,               # 附件4 对齐版本（文件或文件夹）；None 表示自动查找
    VIDEO4_DIR=None,              # 附件4 视频目录；None 表示自动查找
    BERT_PATH=None,               # 本地 bert-base-uncased 目录；None 表示自动查找
    BERT_HUB="bert-base-uncased", # 找不到本地权重时从 HuggingFace 下载（需开 Internet）
    CLS_ID=101, SEP_ID=102, T=50,
    # 随机种子：主配置 5 个，消融 3 个；部署种子事先固定，不按结果挑
    SEEDS_MAIN=[17, 29, 43, 71, 97],
    SEEDS_ABL=[17, 29, 43],
    DEPLOY_SEED=17,
    ABLATIONS=["joint", "mg_nogate", "mg_nokd", "mg_scatter"],
    # 阶段一：LoRA 文本 + 池化双头（只用完整输入）
    S1_EPOCHS=12, S1_BS=32, S1_LR_LORA=2e-4, S1_LR_HEAD=1e-3, S1_WD=0.01,
    LORA_R=8, LORA_ALPHA=16, LORA_DROPOUT=0.05,
    HIDDEN=256, FUSE_DIM=64, DROPOUT=0.2,
    CLASS_WEIGHT="sqrt_inv",      # 类别数平方根倒数加权；None 关闭
    # 阶段二：缺失门控 LoRA
    S2_EPOCHS=10, S2_BS=32, S2_LR=5e-4,
    MISS_R_TEXT=4, MISS_R_FUSE=8, MISS_ALPHA=8,
    KD_T=2.0, LAMBDA_KD=0.5, LAMBDA_FEAT=0.5,
    NOGATE_COMPLETE_P=0.2,        # 无门控消融里保留完整样本的比例
    JOINT_MASK_P=0.5,             # 常规联合训练里缺失样本的比例
    # 训练用缺失族（题面定义：一个或多个模态中的连续区间）
    FAM_WHOLE_P=0.1, FAM_R_MIN=0.05, FAM_R_MAX=0.9, FAM_MAX_SEG=4,   # 每个被选模态 1~4 段连续区间
    # 评估面板
    PANEL_RATIOS=[0.1, 0.3, 0.5],
    SC_RATIOS=[0.1, 0.3, 0.5],
    MS_RATIOS=[0.1, 0.3, 0.5], MS_SEGS=3,   # MS：多段短区间面板
    EXT_RATIOS=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    EVAL_BS=256, BOOT_REPS=1000,
    # 规则层
    RULE_CV_FOLDS=5, RULE_MIN_GAIN=0.003, REVIEW_PMAX=0.5, COHERENT_MAX_MAE_LOSS=0.01,
    # 问题三
    N_EXPLAIN_VALID=96, LOCAL_WIN=3, EVID_TOPK=3,
    FAITH_BUDGETS=[0.1, 0.2, 0.3], FAITH_RAND=20, STAB_NOISE=0.05,
    N_TYPICAL=4,                  # 画逐位置重要性图的典型样本数
    ATT4_TIMING="auto",           # auto：先试 MMS 强制对齐（需 Internet），失败则按词位比例；proportional；off
    ATT4_ALIGN_CSV=None,          # 可选：外部词—秒对齐表（列 sample,position,start_s,end_s）
    # 步骤开关
    RUN_TRAIN=True, RUN_ABLATIONS=False, RUN_EXT_SCAN=True, RUN_RULES=True,
    RUN_ATT3=True, RUN_Q3=True, RUN_TEST=True,
    # ---------------- v2 ----------------
    CKPT_SEARCH=["/kaggle/input", "/kaggle/working"],   # 到这些目录下找 v1 的 .pt（平铺上传也可以）
    MONO_INIT=1.0,                # 单调门控系数 a 的初值（每个缺失量）
    EPOCH0_CANDIDATE=True,        # 阶段二选择时把“不补偿”（epoch 0）也作为候选
    EVAL_CONFIGS=["base", "joint", "mg", "mg2"],
    DEC_W_NEU=[0.5, 3.0, 21], DEC_W_POS=[0.5, 2.0, 15],   # 贝叶斯决策层类别权重网格（对数刻度，w_neg=1）
    DEC_FIT_PANELS=["PM", "SC", "MS"],                     # 缺失权重在这些面板上拟合（LH 只报告不拟合）
    CONF_ALPHA=[0.1, 0.2], SEV_EDGES=[0.0, 0.2, 0.5, 1.0], # 共形：误覆盖率；缺失严重度分桶
    EXT_CONFIGS=["base", "mg", "mg2"], EXT_SPLITS=["valid", "test"],  # 扩展扫描：所有主种子
    REPR_N=400, SVD_N=300, REPR_RATIOS=[0.1, 0.3, 0.5],
    # ---------------- v3：文本主干与训练目标 ----------------
    TEXT_MODEL="microsoft/deberta-v3-large",   # 通用预训练（RTD），没有用任何情感数据微调过
    DEB_PATH=None,                # 本地 DeBERTa 目录；None 表示先在 /kaggle/input 里找，找不到再从 HuggingFace 下载
    DEB_SEEDS=[17, 29, 43, 71, 97],
    DEB_MAXLEN=96, DEB_BS=16, DEB_EPOCHS=10, DEB_LR_LORA=2e-4, DEB_LR_HEAD=1e-3, DEB_WD=0.01, DEB_WARMUP=0.1,
    LORA3_R=16, LORA3_ALPHA=32, LORA3_DROPOUT=0.1,
    LORA3_TARGETS=["query_proj", "key_proj", "value_proj", "attention.output.dense", "intermediate.dense", "output.dense"],
    NGRID=19, ORD_SIGMA=0.25,     # 有序分布头：网格 −3..3（步长 1/3），高斯软标签宽度
    W_ORD=1.0, W_L1=0.5, W_SF1=0.5, RDROP_ALPHA=1.0,
    AUG_P=0.3,                    # 训练时做连续段缺失增广的样本比例（缺失族与 v1/v2 相同，只按题面定义构造）
    DEB_EVAL_BS=128,
    # ---------------- v3：事先锁定的协议 ----------------
    VALID_SPLIT_SALT="gmcm2026E-v3-split", VALID_CAL_FRAC=0.5,   # 验证集按源视频哈希对半分：选择区 VS / 校准区 VC
    TARGETS=dict(f1=0.70, acc=0.70, mae=0.60),
    ENSEMBLES=dict(base_ens=["base"], deb_ens=["deb"], mix_ens=["base", "deb"]),
    V3_EXT_CONFIGS=["base", "joint", "mg", "mg2", "deb"],
    # ---------------- v3：v2 结论补验（只推理） ----------------
    RUN_AUDIT=True, RUN_DEB=True, RUN_PROP3=True, CKPT_ZIP=True,
    AUDIT_CONFIGS=["base", "mg", "joint", "mg2"], AUDIT_RATIOS=[0.75, 0.9, 1.0],
    JOINT_FILL_SEEDS=True,        # joint 在 v1 只有 3 个种子；补训 s71/s97（约 12 分钟），让 joint 对照也有 5 个种子
    AUDIT_RETRAIN_MG2=False,      # 找不到 mg2 检查点时是否重训（s17/s43/s71 选中 epoch 0，本来就与 base 逐位相同）
    PROP3_MAXLAGS=[8, 20],        # 命题 3：自相关估到滞后 8 与 20 各算一次理论值（检验“截断导致理论偏低”）
    # ---------------- v3：附件4 人工核验（优先于工具时间戳） ----------------
    ATT4_MANUAL=[
        dict(sample="15", status="rejected", evidence="all", shift_s=None,
             note="人工核验：文本与音频完全不匹配，不给秒级/帧级定位"),
        dict(sample="02", status="revised", evidence="all", shift_s=None,
             note="人工核验：时间约需修订 0.55 s（方向与对象待填；未填 shift_s 时只标注、不改时间）"),
    ],
    ATT4_MANUAL_CSV=None,         # 可选：外部核验表（列 sample,status,evidence,shift_s,note），会覆盖上面的列表
    ALIGN_NULL_PERM=3,            # 对齐置信度：与打乱词序后的对齐得分比较（重复次数）
)

COMBOS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
POSITIONS = ["head", "mid", "tail"]
MODS = ["T", "A", "V"]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

if CFG["SMOKE"]:
    CFG.update(S1_EPOCHS=1, S2_EPOCHS=1, SEEDS_MAIN=[17], SEEDS_ABL=[17], BOOT_REPS=50, DEB_EPOCHS=1, DEB_SEEDS=[17, 29],
               N_EXPLAIN_VALID=12, FAITH_RAND=3, EXT_RATIOS=[0.5, 1.0], REPR_N=60, SVD_N=60,
               OUT_DIR=CFG["OUT_DIR"] + "_smoke")

OUT = Path(CFG["OUT_DIR"])
for sub in ("", "figures", "ckpt", "runs", "tables", "q3", "att4_keyframes"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
T0 = time.time()
S3 = {}          # v3 运行摘要（报告从这里取数）
S2 = S3          # 兼容 v2 的 guard() 等函数


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    return str(o)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


# %% [data] 1. 数据发现与读取（兼容：整文件 train/valid/test、单样本文件、样本字典、npz）
FEAT_EXT = (".pkl", ".pickle", ".pk", ".npz")
ARRAY_FIELDS = dict(text=3, text_bert=3, audio=3, vision=3, regression_labels=1, classification_labels=1,
                    audio_lengths=1, vision_lengths=1)
STR_FIELDS = ("id", "raw_text", "annotations")
ATT_KEYS = dict(att2=["附件2", "attachment2", "attachment_2", "att2"],
                att3=["附件3", "attachment3", "attachment_3", "att3"],
                att4=["附件4", "attachment4", "attachment_4", "att4"])


def is_unaligned(p):
    return ("unaligned" in p.lower()) or ("未对齐" in p)


def att_of(p):
    low = p.lower()
    hit = [k for k, ws in ATT_KEYS.items() if any(w in low for w in ws)]
    return hit[0] if len(hit) == 1 else None


class _NumpyCompatUnpickler(pickle.Unpickler):
    """附件4 的 pkl 是用 numpy 2.x 存的（模块路径 numpy._core.*），Kaggle 镜像是 numpy 1.26（numpy.core.*）。"""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            try:
                __import__(module)
            except ImportError:
                module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_obj(path):
    if path.lower().endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        d = {}
        for k in z.files:
            v = z[k]
            d[k] = v.item() if (v.dtype == object and v.ndim == 0) else v
        return d
    with open(path, "rb") as f:
        try:
            return _NumpyCompatUnpickler(f).load()
        except UnicodeDecodeError:
            f.seek(0)
            return _NumpyCompatUnpickler(f, encoding="latin1").load()


def _is_field_dict(d):
    return isinstance(d, dict) and any(k in d for k in ("audio", "vision", "text_bert"))


def _fix_dims(k, arr, single):
    arr = np.asarray(arr)
    want = ARRAY_FIELDS[k]
    if want == 1:
        return arr.reshape(-1)
    if single and arr.ndim == want - 1:
        arr = arr[None]
    while arr.ndim > want and arr.shape[1] == 1:       # 列表里每个元素是 (1, ...)：挤掉多余的 1
        arr = arr[:, 0]
    return arr


def _as_batch(d, default_id):
    key = next(k for k in ("audio", "vision", "text_bert") if k in d)
    try:
        single = np.asarray(d[key]).ndim == 2           # 单样本文件里存成 (50,74) / (3,50) 的情形
    except Exception:
        single = False
    out = {}
    for k, v in d.items():
        if k in ARRAY_FIELDS:
            try:
                out[k] = _fix_dims(k, v, single)
            except Exception:
                pass
        elif k in STR_FIELDS:
            a = np.asarray(v, dtype=object).reshape(-1)
            out[k] = [x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x) for x in a]
    n = len(out["audio"]) if "audio" in out else len(out["text_bert"])
    if "id" not in out or len(out["id"]) != n:
        out["id"] = [str(default_id)] if n == 1 else [f"{default_id}_{i:03d}" for i in range(n)]
    for k in ("raw_text", "annotations"):
        if k in out and len(out[k]) != n:
            del out[k]
    return out


def _concat_batches(parts):
    parts = [p for p in parts if p]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    out = {}
    for k in keys:
        if k in STR_FIELDS:
            out[k] = sum([list(p[k]) for p in parts], [])
        else:
            try:
                out[k] = np.concatenate([np.asarray(p[k]) for p in parts], 0)
            except Exception:
                pass
    return out


def extract_batches(obj, name):
    """任意结构 → {划分名: 字段字典}。划分名为 train/valid/test，或 all。"""
    if isinstance(obj, dict):
        sk = [k for k in ("train", "valid", "test") if k in obj and isinstance(obj[k], dict)]
        if sk:
            return {k: _as_batch(obj[k], f"{name}_{k}") for k in sk}
        if _is_field_dict(obj):
            return {"all": _as_batch(obj, name)}
        if len(obj) == 1:
            return extract_batches(next(iter(obj.values())), name)
        parts = []
        for k in sorted(obj.keys(), key=natural_key):
            v = obj[k]
            if isinstance(v, (dict, list, tuple)):
                parts += list(extract_batches(v, k).values())
        return {"all": _concat_batches(parts)} if parts else {}
    if isinstance(obj, (list, tuple)) and len(obj) and isinstance(obj[0], dict):
        parts = []
        for i, o in enumerate(obj):
            parts += list(extract_batches(o, f"{name}_{i:03d}").values())
        return {"all": _concat_batches(parts)}
    return {}


def _T_of(b):
    return np.asarray(b["audio"]).shape[1] if "audio" in b else np.asarray(b["text_bert"]).shape[-1]


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def load_folder(paths, name):
    """一个附件下的多个特征文件 → 一个字段字典（按自然顺序拼接）。"""
    parts = []
    for p in sorted(paths, key=natural_key):
        try:
            obj = load_obj(p)
        except Exception as e:
            log("   读取失败", p, e); continue
        bs = extract_batches(obj, _stem(p))
        for k, b in bs.items():
            if not b:
                continue
            if _T_of(b) != CFG["T"]:
                log(f"   跳过 {p}（序列长度 {_T_of(b)}，不是对齐版 {CFG['T']}）"); continue
            parts.append(b)
        del obj
    return _concat_batches(parts)


def describe_batch(b):
    return {k: (np.asarray(v).shape if k not in STR_FIELDS else f"list[{len(v)}]") for k, v in b.items()}


_ATT2_CACHE = {}


def discover_inputs():
    root = CFG["INPUT_ROOT"]
    files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(FEAT_EXT) and os.path.isfile(p)]
    n_un = sum(is_unaligned(p) for p in files)
    files = [p for p in files if not is_unaligned(p)]
    log(f"在 {root} 下找到 {len(files) + n_un} 个特征文件（其中未对齐版 {n_un} 个，已跳过）：")
    for p in sorted(files, key=natural_key)[:60]:
        log(f"   {p}  {os.path.getsize(p) / 1e6:.2f} MB")
    if len(files) > 60:
        log(f"   ……共 {len(files)} 个")
    groups = {k: [] for k in ATT_KEYS}
    for p in files:
        a = att_of(p)
        if a:
            groups[a].append(p)
    # 附件2：优先手动指定；否则在附件2目录里找含 train/valid/test 的对齐文件
    att2 = CFG["ATT2_PATH"]
    if att2 is None:
        cands = groups["att2"] or files
        cands = sorted(cands, key=lambda p: (0 if "aligned_50" in os.path.basename(p).lower() else 1, -os.path.getsize(p)))
        for p in cands:
            try:
                obj = load_obj(p)
            except Exception:
                continue
            bs = extract_batches(obj, _stem(p))
            del obj
            if "train" in bs and _T_of(bs["train"]) == CFG["T"]:
                att2 = p
                _ATT2_CACHE[p] = bs
                break
            del bs
        if att2 is None:
            raise FileNotFoundError("没有找到附件2 对齐版（含 train/valid/test 且长度 50）。请设置 CFG['ATT2_PATH']")

    def pick(key, manual):
        if manual:
            if os.path.isdir(manual):
                return [p for p in glob.glob(os.path.join(manual, "**", "*"), recursive=True)
                        if p.lower().endswith(FEAT_EXT) and not is_unaligned(p)]
            return [manual]
        return [p for p in groups[key] if p != att2]

    att3 = pick("att3", CFG["ATT3_PATH"]); att4 = pick("att4", CFG["ATT4_PATH"])
    log("附件2：", att2)
    att3 = sorted(att3, key=natural_key); att4 = sorted(att4, key=natural_key)
    log(f"附件3：{len(att3)} 个文件", [os.path.basename(p) for p in att3[:3]], "..." if len(att3) > 3 else "")
    log(f"附件4：{len(att4)} 个文件", [os.path.basename(p) for p in att4[:3]], "..." if len(att4) > 3 else "")
    return att2, att3, att4


def find_videos4():
    if CFG["VIDEO4_DIR"]:
        roots = [CFG["VIDEO4_DIR"]]
    else:
        roots = [CFG["INPUT_ROOT"]]
    vids = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "*"), recursive=True):
            if not p.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            if CFG["VIDEO4_DIR"] or (att_of(p) == "att4" and not is_unaligned(p)):
                vids.append(p)
    return sorted(vids, key=natural_key)


def find_bert():
    if CFG["BERT_PATH"]:
        return CFG["BERT_PATH"]
    for cfgp in glob.glob(os.path.join(CFG["INPUT_ROOT"], "**", "config.json"), recursive=True):
        d = os.path.dirname(cfgp)
        fs = os.listdir(d)
        if "vocab.txt" not in fs or not any(f in fs for f in ("pytorch_model.bin", "model.safetensors")):
            continue
        try:
            c = json.load(open(cfgp))
        except Exception:
            continue
        if c.get("model_type") == "bert" and c.get("hidden_size") == 768 and c.get("vocab_size") == 30522:
            log("使用本地 BERT：", d)
            return d
    log("没找到本地 BERT，改用 HuggingFace：", CFG["BERT_HUB"])
    return CFG["BERT_HUB"]


def find_deberta():
    """先找本地 DeBERTa-v3（config.json 里 model_type=deberta-v2 且带 spm.model），找不到再用 HuggingFace 名称下载。"""
    if CFG["DEB_PATH"]:
        return CFG["DEB_PATH"]
    want_h = 1024 if "large" in CFG["TEXT_MODEL"] else None
    for cfgp in glob.glob(os.path.join(CFG["INPUT_ROOT"], "**", "config.json"), recursive=True):
        d = os.path.dirname(cfgp)
        fs = os.listdir(d)
        if "spm.model" not in fs or not any(f in fs for f in ("pytorch_model.bin", "model.safetensors")):
            continue
        try:
            c = json.load(open(cfgp))
        except Exception:
            continue
        if c.get("model_type") == "deberta-v2" and (want_h is None or c.get("hidden_size") == want_h):
            log("使用本地 DeBERTa：", d)
            return d
    log("没找到本地 DeBERTa，改用 HuggingFace：", CFG["TEXT_MODEL"])
    return CFG["TEXT_MODEL"]


def _split_text_bert(tb):
    """text_bert → (input_ids, attention_mask, token_type_ids)，自动识别通道顺序。"""
    tb = np.asarray(tb)
    if tb.ndim == 3 and tb.shape[1] != 3 and tb.shape[2] == 3:
        tb = tb.transpose(0, 2, 1)
    tb = np.rint(tb).astype(np.int64)
    chans = [tb[:, c, :] for c in range(3)]
    i_ids = int(np.argmax([c.max() for c in chans]))
    rest = [c for c in range(3) if c != i_ids]
    i_mask = rest[int(np.argmax([chans[c].sum() for c in rest]))]
    i_tt = [c for c in rest if c != i_mask][0]
    return chans[i_ids].copy(), chans[i_mask].copy(), chans[i_tt].copy()


def _longest_run(b):
    n, L = b.shape
    best = np.zeros(n, dtype=np.int64); cur = np.zeros(n, dtype=np.int64)
    for t in range(L):
        cur = np.where(b[:, t], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _last_true(b):
    """每行最后一个 True 的位置，没有则 -1。"""
    L = b.shape[1]
    has = b.any(1)
    return np.where(has, L - 1 - np.argmax(b[:, ::-1], 1), -1)


def _first_true(b):
    has = b.any(1)
    return np.where(has, np.argmax(b, 1), -1)


_UID = [0]


class MMData:
    """一个划分（或一个专项测试集）的全部数组与掩码。
    词范围 W = 位置 1..sep-1；文本缺失以 W 为参照；声学/视觉缺失以参照区间 R_A、R_V 为参照
    （R = W 按训练集估计的整体偏移平移，正常数据下偏移为 0）。"""

    def __init__(self, d, name, has_label=True):
        _UID[0] += 1; self.uid = _UID[0]
        self.name = name
        audio = np.asarray(d["audio"], dtype=np.float32)
        vision = np.asarray(d["vision"], dtype=np.float32)
        n, T = audio.shape[0], audio.shape[1]
        assert T == CFG["T"], f"{name}: 序列长度为 {T}，本流程只支持对齐版 aligned_50"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        vision = np.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)
        self.n, self.T = n, T
        self.ids = [str(x) for x in d["id"]]
        self.videos = [s.split("$_$")[0] for s in self.ids]
        if "text_bert" not in d:
            raise KeyError(f"{name} 缺少 text_bert 字段（本流程在线编码文本，需要词元编号）")
        ids, attn, self.token_type = _split_text_bert(d["text_bert"])
        self.raw_text = [str(x) for x in d["raw_text"]] if "raw_text" in d else [""] * n
        nz_text = None
        if "text" in d and d["text"] is not None:
            tx = np.asarray(d["text"], dtype=np.float32)
            if tx.ndim == 3 and tx.shape[1] == T:
                nz_text = np.linalg.norm(tx, axis=2) > 1e-8
            del tx
        nzA = np.linalg.norm(audio, axis=2) > 1e-8
        nzV = np.linalg.norm(vision, axis=2) > 1e-8
        # 结构词元：[CLS] 被置零的补回；找不到 [SEP] 时用 “词元/注意力/声学/视觉” 中最靠后的非零位置推断句长
        self.fix = dict(cls_restored=0, sep_restored=0)
        miss_cls = ids[:, 0] == 0
        ids[miss_cls, 0] = CFG["CLS_ID"]; self.fix["cls_restored"] = int(miss_cls.sum())
        sep = np.full(n, -1, dtype=np.int64)
        last_any = np.max(np.stack([_last_true(ids > 0), _last_true(attn > 0), _last_true(nzA), _last_true(nzV)]), 0)
        for i in range(n):
            w = np.flatnonzero(ids[i] == CFG["SEP_ID"])
            if len(w):
                sep[i] = w[0]
            else:
                sep[i] = int(np.clip(last_any[i] + 1, 1, T - 1))
                if ids[i, sep[i]] == 0:
                    ids[i, sep[i]] = CFG["SEP_ID"]; attn[i, sep[i]] = 1
                self.fix["sep_restored"] += 1
        self.input_ids = ids
        self.sep = sep
        pos = np.arange(T)[None, :]
        self.W = (pos >= 1) & (pos < sep[:, None])
        tmiss = self.W & ((ids == 0) | (attn == 0))
        if nz_text is not None:
            tmiss |= self.W & (~nz_text)
        self.oT = self.W & ~tmiss
        attn = attn.copy(); attn[tmiss] = 0; attn[:, 0] = 1; attn[np.arange(n), sep] = 1
        self.attn = attn
        self.oA, self.oV = nzA, nzV
        self.audio_raw, self.vision_raw = audio, vision
        self.audio = self.vision = None
        self.RT = self.W
        self.RA = self.RV = self.W
        self.has_label = has_label and ("regression_labels" in d)
        self.label_check = {}
        if self.has_label:
            y = np.asarray(d["regression_labels"], dtype=np.float32).reshape(-1)
            self.y = np.clip(y, -3, 3)
            self.c = np.where(y < 0, 0, np.where(y > 0, 2, 1)).astype(np.int64)
            if "annotations" in d:
                m = {"negative": 0, "neutral": 1, "positive": 2}
                a = np.array([m.get(str(x).strip().lower(), -1) for x in d["annotations"]])
                ok = a >= 0
                self.label_check["annotations_agree"] = float((a[ok] == self.c[ok]).mean()) if ok.any() else None
            if "classification_labels" in d:
                cl = np.asarray(d["classification_labels"]).reshape(-1)
                for tag, mp in (("cls_0_1_2", {0: 0, 1: 1, 2: 2}), ("cls_neg1_0_1", {-1: 0, 0: 1, 1: 2})):
                    a = np.array([mp.get(int(round(float(x))), -1) for x in cl])
                    ok = a >= 0
                    if ok.mean() > 0.99:
                        self.label_check[f"classification_labels_agree_{tag}"] = float((a[ok] == self.c[ok]).mean())
        else:
            self.y = None; self.c = None

    def set_ref(self, offs):
        pos = np.arange(self.T)[None, :]
        for m, attr in (("A", "RA"), ("V", "RV")):
            d0, d1 = offs[m]
            lo = 1 + d0; hi = self.sep[:, None] - 1 + d1
            setattr(self, attr, (pos >= lo) & (pos <= hi))

    def o(self, m):
        return {"T": self.oT, "A": self.oA, "V": self.oV}[m]

    def R(self, m):
        return {"T": self.RT, "A": self.RA, "V": self.RV}[m]

    def standardize(self, stats):
        (muA, sdA), (muV, sdV) = stats
        self.audio = ((self.audio_raw - muA) / sdA * self.oA[..., None]).astype(np.float32)
        self.vision = ((self.vision_raw - muV) / sdV * self.oV[..., None]).astype(np.float32)

    def subset(self, idx, name=None):
        s = copy.copy(self)
        _UID[0] += 1; s.uid = _UID[0]
        idx = np.asarray(idx)
        for k in ("input_ids", "token_type", "attn", "sep", "W", "oT", "oA", "oV", "RT", "RA", "RV",
                  "audio_raw", "vision_raw"):
            setattr(s, k, getattr(self, k)[idx])
        if self.audio is not None:
            s.audio = self.audio[idx]; s.vision = self.vision[idx]
        for k in getattr(self, "row_extra", ()):         # v3：DeBERTa 词元与映射数组（有才切）
            setattr(s, k, getattr(self, k)[idx])
        s.ids = [self.ids[i] for i in idx]; s.videos = [self.videos[i] for i in idx]
        s.raw_text = [self.raw_text[i] for i in idx]
        if self.has_label:
            s.y = self.y[idx]; s.c = self.c[idx]
        s.n = len(idx); s.name = name or self.name
        return s


def estimate_offsets(tr):
    """声学/视觉的首末非零位置相对词范围 [1, sep-1] 的典型偏移（众数）。正常对齐数据应为 (0, 0)。"""
    offs, info = {}, {}
    for m in ("A", "V"):
        o = tr.o(m)
        f, l = _first_true(o), _last_true(o)
        ok = f >= 0
        d0 = (f - 1)[ok]; d1 = (l - (tr.sep - 1))[ok]
        res = []
        for d in (d0, d1):
            c = Counter(d.tolist()).most_common(1)
            val, share = (c[0][0], c[0][1] / max(1, len(d))) if c else (0, 0)
            res.append((int(val) if share >= 0.5 else 0, float(share)))
        offs[m] = (res[0][0], res[1][0])
        info[m] = dict(first_offset_mode=res[0][0], first_share=res[0][1], last_offset_mode=res[1][0], last_share=res[1][1])
    log("声学/视觉参照区间偏移（相对词范围）：", info)
    return offs, info


def train_stats(ds):
    out = []
    for x, o in ((ds.audio_raw, ds.oA), (ds.vision_raw, ds.oV)):
        v = x[o]
        mu = v.mean(0) if len(v) else np.zeros(x.shape[2], np.float32)
        sd = v.std(0) if len(v) else np.ones(x.shape[2], np.float32)
        sd = np.where(sd < 1e-6, 1.0, sd)
        out.append((mu.astype(np.float32), sd.astype(np.float32)))
    return out


def load_all():
    p2, p3, p4 = discover_inputs()
    bs = _ATT2_CACHE.pop(p2, None) or extract_batches(load_obj(p2), "att2")
    log("附件2 结构：", {k: describe_batch(v) for k, v in bs.items()})
    data = {}
    for k in ("train", "valid", "test"):
        if k in bs:
            data[k] = MMData(bs[k], k, has_label=True)
    del bs
    for key, paths in (("att3", p3), ("att4", p4)):
        if not paths:
            log(f"未找到 {key}，相关步骤将跳过"); continue
        b = load_folder(paths, key)
        if not b:
            log(f"{key} 读取为空，相关步骤将跳过"); continue
        log(f"{key} 结构：", describe_batch(b))
        data[key] = MMData(b, key, has_label=False)
    offs, off_info = estimate_offsets(data["train"])
    for ds in data.values():
        ds.set_ref(offs)
    stats = train_stats(data["train"])
    for ds in data.values():
        ds.standardize(stats)
    if CFG["SMOKE"]:
        rng = np.random.default_rng(0)
        data["train"] = data["train"].subset(np.sort(rng.choice(data["train"].n, min(256, data["train"].n), replace=False)))
        for k in ("valid", "test"):
            if k in data:
                data[k] = data[k].subset(np.arange(min(96, data[k].n)))
    return data, stats, off_info


def audit(data, off_info=None):
    rows = []
    for k, ds in data.items():
        r = dict(split=k, n=ds.n, words_mean=float(ds.W.sum(1).mean()), sep_restored=ds.fix["sep_restored"],
                 cls_restored=ds.fix["cls_restored"])
        for m in MODS:
            o, R = ds.o(m), ds.R(m)
            miss = R & ~o
            r[f"{m}_obs_mean"] = float(o.sum(1).mean())
            r[f"{m}_samples_with_missing"] = int((miss.sum(1) > 0).sum())
            r[f"{m}_missing_frac_mean"] = float((miss.sum(1) / np.maximum(1, R.sum(1))).mean())
            r[f"{m}_longest_gap_mean"] = float(_longest_run(miss).mean())
            r[f"{m}_empty_samples"] = int((o.sum(1) == 0).sum())
            r[f"{m}_obs_outside_ref"] = int((o & ~R).sum())
        if ds.has_label:
            cnt = np.bincount(ds.c, minlength=3)
            r.update(neg=int(cnt[0]), neu=int(cnt[1]), pos=int(cnt[2]), y_mean=float(ds.y.mean()))
            r.update(ds.label_check)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "data_audit.csv", index=False)
    if off_info:
        save_json(off_info, OUT / "tables" / "ref_offsets.json")
    log("数据审计：\n" + df.T.to_string())
    return df

# %% [masks] 2. 缺失掩码：训练缺失族、PM 面板、LH 压力面板、SC 散点面板、扩展扫描
def _empty_masks(n, T):
    return {m: np.zeros((n, T), dtype=bool) for m in MODS}


def window_mask(o_row, r, pos, allow_clear=False):
    """在一行可观测位置上按“名义比例 r、头/中/尾”截取一段连续窗口（与第二篇的 PM 规则一致）。"""
    s = np.zeros_like(o_row)
    P = np.flatnonzero(o_row)
    if len(P) == 0:
        return s
    if r >= 1.0 and allow_clear:
        s[P] = True
        return s
    if len(P) < 2:
        return s
    f, l = P[0], P[-1]
    L = l - f + 1
    w = min(max(1, int(np.floor(r * L))), L)
    st = f if pos == "head" else (f + (L - w) // 2 if pos == "mid" else l - w + 1)
    s[st:st + w] = True
    s &= o_row
    if not allow_clear and s.sum() >= o_row.sum():      # 不把原本可观测的模态整个删空
        s[P[-1] if pos != "tail" else P[0]] = False
    return s


def build_panel(ds, kind="PM", ratios=None, allow_clear=False):
    """返回 [(bucket_name, combo, ratio, pos, masks_dict)]。
    PM：被遮蔽模态内删一段连续窗口，其余模态不动；
    LH：只保留子集 S，S 以外的模态整路关闭，S 内再删一段连续窗口。"""
    ratios = ratios or CFG["PANEL_RATIOS"]
    cache = {}

    def win(m, r, pos, clear):
        key = (m, r, pos, clear)
        if key not in cache:
            o = ds.o(m)
            cache[key] = np.stack([window_mask(o[i], r, pos, clear) for i in range(ds.n)]) if ds.n else \
                np.zeros((0, ds.T), dtype=bool)
        return cache[key]

    zero = np.zeros((ds.n, ds.T), dtype=bool)
    out = []
    for combo in COMBOS:
        for r in ratios:
            for pos in POSITIONS:
                ms = {}
                for m in MODS:
                    if kind == "PM":
                        ms[m] = win(m, r, pos, allow_clear) if m in combo else zero
                    else:
                        ms[m] = win(m, r, pos, False) if m in combo else ds.o(m)
                out.append((f"{kind}|{combo}|{r}|{pos}", combo, r, pos, ms))
    return out


def build_sc_panel(ds, ratios=None, seed=2026):
    """SC：训练分布外的“散点缺失”——被选模态的每个可观测位置独立以概率 r 丢弃（至少保留 1 个）。"""
    ratios = ratios or CFG["SC_RATIOS"]
    rng = np.random.default_rng(seed)
    zero = np.zeros((ds.n, ds.T), dtype=bool)
    base = {}
    for m in MODS:
        u = rng.random((ds.n, ds.T))
        base[m] = u
    out = []
    for combo in COMBOS:
        for r in ratios:
            ms = {}
            for m in MODS:
                if m not in combo:
                    ms[m] = zero; continue
                o = ds.o(m)
                s = (base[m] < r) & o
                full = s.sum(1) >= o.sum(1)
                for i in np.flatnonzero(full & (o.sum(1) > 0)):
                    s[i, np.flatnonzero(o[i])[0]] = False
                ms[m] = s
            out.append((f"SC|{combo}|{r}|scatter", combo, r, "scatter", ms))
    return out


def build_ms_panel(ds, ratios=None, n_seg=None, seed=2027):
    """MS：多段短区间——被选模态里放 n_seg 段互不重叠的连续区间，总长约为比例 r（每段至少 1 个位置）。"""
    ratios = ratios or CFG["MS_RATIOS"]; n_seg = n_seg or CFG["MS_SEGS"]
    rng = np.random.default_rng(seed)
    zero = np.zeros((ds.n, ds.T), dtype=bool)
    out = []
    for combo in COMBOS:
        for r in ratios:
            ms = {}
            for m in MODS:
                if m not in combo:
                    ms[m] = zero; continue
                o = ds.o(m); s = np.zeros_like(o)
                for i in range(ds.n):
                    P = np.flatnonzero(o[i])
                    if len(P) < 2:
                        continue
                    total = min(max(n_seg, int(round(r * len(P)))), len(P) - 1)
                    k = min(n_seg, total)
                    lens = np.full(k, total // k); lens[: total % k] += 1
                    free = np.ones(len(P), bool)
                    for ln in lens:
                        cand = [j for j in range(len(P) - ln + 1) if free[max(0, j - 1):j + ln + 1].all()]
                        if not cand:
                            break
                        j = cand[rng.integers(len(cand))]
                        s[i, P[j:j + ln]] = True; free[j:j + ln] = False
                ms[m] = s
            out.append((f"MS|{combo}|{r}|multi", combo, r, "multi", ms))
    return out


def sample_family_masks(ds, idx, rng, p_masked=1.0, family="contig"):
    """训练用缺失族。contig：按题面定义（“一个或多个模态中存在随机连续序列区间”），
    在随机选中的模态里删除 1~FAM_MAX_SEG 段互不重叠的连续区间，总比例 U(FAM_R_MIN, FAM_R_MAX)，小概率整路删除；
    scatter（消融用）：删除比例同分布，但位置逐点独立，不成段。两者都只用题面的文字定义，没有参考附件3。"""
    n = len(idx)
    ms = _empty_masks(n, ds.T)
    masked = np.zeros(n, dtype=bool)
    for i, j in enumerate(idx):
        if rng.random() >= p_masked:
            continue
        combo = COMBOS[rng.integers(len(COMBOS))]
        for m in combo:
            o = ds.o(m)[j]
            P = np.flatnonzero(o)
            if len(P) == 0:
                continue
            if rng.random() < CFG["FAM_WHOLE_P"]:
                ms[m][i, P] = True; masked[i] = True
                continue
            if len(P) < 2:
                continue
            r = rng.uniform(CFG["FAM_R_MIN"], CFG["FAM_R_MAX"])
            if family == "scatter":
                s = np.zeros(ds.T, dtype=bool)
                s[P[rng.random(len(P)) < r]] = True
                if s.sum() >= len(P):
                    s[P[0]] = False
                ms[m][i] = s
                masked[i] |= s.any()
                continue
            f, l = P[0], P[-1]
            L = l - f + 1
            total = max(1, int(round(r * L)))
            k = int(rng.integers(1, CFG["FAM_MAX_SEG"] + 1))
            k = max(1, min(k, total))
            cuts = np.sort(rng.choice(np.arange(1, total), k - 1, replace=False)) if k > 1 else np.array([], int)
            lens = np.diff(np.concatenate([[0], cuts, [total]])).astype(int)
            used = np.zeros(ds.T, dtype=bool)
            for ln in rng.permutation(lens):
                for _ in range(20):
                    st = int(rng.integers(f, max(f, l - ln + 1) + 1))
                    seg = np.zeros(ds.T, dtype=bool); seg[st:st + ln] = True
                    grow = seg.copy(); grow[max(0, st - 1):st + ln + 1] = True      # 段与段之间至少隔 1 个位置
                    if not (grow & used).any():
                        used |= seg
                        break
            ms[m][i] = used & o
            masked[i] |= ms[m][i].any()
    return ms, masked


def ref_rows(ds, idx):
    return ds.RT[idx], ds.RA[idx], ds.RV[idx]


def descriptors(aT, aA, aV, R):
    """缺失描述量 φ（9 维）：每个模态 [缺失比例, 最长缺口比例, 是否有缺失]，以各自参照区间为分母。"""
    feats, anys = [], []
    for a, Rm in zip((aT, aA, aV), R):
        Rn = np.maximum(1, Rm.sum(1))
        miss = Rm & ~a
        frac = miss.sum(1) / Rn
        gap = _longest_run(miss) / Rn
        anym = (miss.sum(1) > 0).astype(np.float32)
        feats += [frac, gap, anym]
        anys.append(anym)
    phi = np.stack(feats, 1).astype(np.float32)
    indT = anys[0]
    indF = np.clip(anys[0] + anys[1] + anys[2], 0, 1).astype(np.float32)
    return phi, indT, indF


def apply_masks(ds, idx, ms=None):
    """原生可观测掩码 − 删除掩码 = 实际可用掩码 a；文本删除位置的注意力置 0。"""
    oT, oA, oV = ds.oT[idx], ds.oA[idx], ds.oV[idx]
    attn = ds.attn[idx].copy()
    if ms is not None:
        aT = oT & ~ms["T"]; aA = oA & ~ms["A"]; aV = oV & ~ms["V"]
        attn[ms["T"] & ds.W[idx]] = 0
    else:
        aT, aA, aV = oT, oA, oV
    attn[:, 0] = 1
    return aT, aA, aV, attn


# %% [metrics] 3. 指标与视频配对重抽样
def prior_of(tr):
    """全部模态都缺失时的兜底输出：训练集类别频率与强度中位数。"""
    p = np.bincount(tr.c, minlength=3) / tr.n
    return p.astype(np.float32), float(np.median(tr.y))


def conf_mat(c_true, c_pred, w=None):
    return np.bincount(c_true * 3 + c_pred, weights=w, minlength=9).reshape(3, 3)


def f1_from_cm(C):
    f1 = []
    for k in range(3):
        tp = C[k, k]; fp = C[:, k].sum() - tp; fn = C[k, :].sum() - tp
        d = 2 * tp + fp + fn
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f1)), f1


def _weighted_f1_binary(t, p):
    out, tot = 0.0, len(t)
    for k in (0, 1):
        tp = np.sum((p == k) & (t == k)); fp = np.sum((p == k) & (t != k)); fn = np.sum((p != k) & (t == k))
        d = 2 * tp + fp + fn
        out += (np.sum(t == k) / max(1, tot)) * (0.0 if d == 0 else 2 * tp / d)
    return float(out)


def ece_score(probs, c_true, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == c_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(c_true, y_true, probs, yhat, c_pred=None):
    if c_pred is None:
        c_pred = probs.argmax(1)
    C = conf_mat(c_true, c_pred)
    f1, per = f1_from_cm(C)
    corr = np.nan
    if np.std(yhat) > 1e-9 and np.std(y_true) > 1e-9:
        corr = float(np.corrcoef(yhat, y_true)[0, 1])
    rec = [C[k, k] / C[k].sum() if C[k].sum() else 0.0 for k in range(3)]
    prec = [C[k, k] / C[:, k].sum() if C[:, k].sum() else 0.0 for k in range(3)]
    nz = y_true != 0
    t2 = (y_true[nz] > 0).astype(int); p2 = (yhat[nz] > 0).astype(int)
    y7 = np.clip(np.round(y_true), -3, 3); p7 = np.clip(np.round(yhat), -3, 3)
    y5 = np.clip(np.round(y_true), -2, 2); p5 = np.clip(np.round(yhat), -2, 2)
    return dict(acc=float((c_pred == c_true).mean()), f1=f1, mae=float(np.abs(yhat - y_true).mean()), corr=corr,
                f1_neg=per[0], f1_neu=per[1], f1_pos=per[2], rec_neg=rec[0], rec_neu=rec[1], rec_pos=rec[2],
                prec_neu=prec[1],
                acc2=float((t2 == p2).mean()) if nz.any() else np.nan,
                f1_2w=_weighted_f1_binary(t2, p2) if nz.any() else np.nan,
                acc5=float((y5 == p5).mean()), acc7=float((y7 == p7).mean()),
                ece=ece_score(probs, c_true))


def panel_table(ds, panel_preds):
    rows = []
    for name, combo, r, pos, pr in panel_preds:
        m = metrics(ds.c, ds.y, pr["probs"], pr["yhat"])
        m.update(bucket=name, combo=combo, ratio=r, pos=pos, gateF=float(pr["gF"].mean()), gateT=float(pr["gT"].mean()))
        rows.append(m)
    return pd.DataFrame(rows)


def panel_summary(df, complete=None):
    s = dict(avg_f1=float(df.f1.mean()), avg_acc=float(df.acc.mean()), avg_mae=float(df.mae.mean()),
             avg_corr=float(df["corr"].mean()), avg_acc2=float(df.acc2.mean()), avg_ece=float(df.ece.mean()),
             worst_f1=float(df.f1.min()), worst_bucket=df.loc[df.f1.idxmin(), "bucket"], worst_mae=float(df.mae.max()))
    if complete is not None:
        s["gap_f1"] = complete["f1"] - s["avg_f1"]
        s["gap_mae"] = s["avg_mae"] - complete["mae"]
        s["gap_worst_f1"] = complete["f1"] - s["worst_f1"]
    return s


def paired_bootstrap(ds, predsA, predsB, reps=None, seed=0):
    """按源视频有放回抽样，比较两组 bucket 预测（同一批样本）的平均 F1、平均 MAE 与最差档 F1。"""
    reps = reps or CFG["BOOT_REPS"]
    rng = np.random.default_rng(seed)
    uv, inv = np.unique(np.array(ds.videos), return_inverse=True)
    G = len(uv)
    cA = [p["probs"].argmax(1) if "c" not in p else p["c"] for p in predsA]
    cB = [p["probs"].argmax(1) if "c" not in p else p["c"] for p in predsB]
    yA = [p["yhat"] for p in predsA]; yB = [p["yhat"] for p in predsB]

    def stat(w):
        fA = [f1_from_cm(conf_mat(ds.c, c, w))[0] for c in cA]
        fB = [f1_from_cm(conf_mat(ds.c, c, w))[0] for c in cB]
        ww = np.ones(ds.n) if w is None else w
        mA = [np.sum(ww * np.abs(y - ds.y)) / ww.sum() for y in yA]
        mB = [np.sum(ww * np.abs(y - ds.y)) / ww.sum() for y in yB]
        aA = [np.sum(ww * (c == ds.c)) / ww.sum() for c in cA]
        aB = [np.sum(ww * (c == ds.c)) / ww.sum() for c in cB]
        return [np.mean(fA) - np.mean(fB), np.mean(mA) - np.mean(mB), np.min(fA) - np.min(fB), np.mean(aA) - np.mean(aB)]

    diffs = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(np.float64)
        if w.sum() > 0:
            diffs.append(stat(w))
    d = np.array(diffs); pt = stat(None)
    out = {}
    for k, name in enumerate(["avg_f1", "avg_mae", "worst_f1", "avg_acc"]):
        out[name] = dict(point=float(pt[k]), lo=float(np.percentile(d[:, k], 2.5)), hi=float(np.percentile(d[:, k], 97.5)))
    out["groups"] = int(G)
    return out


# %% [ens] 3b. v3 · 有序分布、集成、验证集对半切分、按视频的区间
GRID = np.linspace(-3.0, 3.0, 19)              # 标注格点：MOSEI 强度是 3 位标注者的均值，落在 1/3 的格点上


def dist_median(dist):
    """分布 → 中位数（MAE 最优点估计）。每个格点的质量在 [g−h/2, g+h/2] 上均匀摊开，于是中位数连续可插值。"""
    dist = np.asarray(dist, np.float64)
    dist = dist / np.maximum(dist.sum(1, keepdims=True), 1e-12)
    h = GRID[1] - GRID[0]
    cdf = np.cumsum(dist, 1)
    k = np.minimum((cdf < 0.5).sum(1), dist.shape[1] - 1)
    prev = np.where(k > 0, cdf[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
    pk = np.maximum(dist[np.arange(len(k)), k], 1e-12)
    m = GRID[k] - h / 2 + (0.5 - prev) / pk * h
    return np.clip(m, -3, 3).astype(np.float32)


def dist_to3(dist):
    """分布 → 三类概率（格点 <0 / =0 / >0）。只作诊断，不参与决策。"""
    dist = np.asarray(dist)
    return np.stack([dist[:, GRID < -1e-9].sum(1), dist[:, np.abs(GRID) < 1e-9].sum(1), dist[:, GRID > 1e-9].sum(1)], 1)


def combine_preds(parts):
    """集成（事先定好的规则）：类别概率取平均；成员都有强度分布时平均分布再取中位数，否则平均点估计。"""
    out = dict(probs=np.mean([p["probs"] for p in parts], 0).astype(np.float32),
               gT=np.mean([p["gT"] for p in parts], 0).astype(np.float32),
               gF=np.mean([p["gF"] for p in parts], 0).astype(np.float32))
    if all("dist" in p for p in parts):
        out["dist"] = np.mean([p["dist"] for p in parts], 0).astype(np.float32)
        out["yhat"] = dist_median(out["dist"])
    else:
        out["yhat"] = np.mean([p["yhat"] for p in parts], 0).astype(np.float32)
    out["member_yhat_sd"] = np.std([p["yhat"] for p in parts], 0).astype(np.float32)
    out["member_agree"] = np.mean([p["probs"].argmax(1) == out["probs"].argmax(1) for p in parts], 0).astype(np.float32)
    return out


class EnsemblePredictor:
    """把若干个成员（TorchPredictor 或别的集成）包装成同一个预测接口。"""

    def __init__(self, members, name="ens"):
        self.members = list(members); self.name = name
        self.gate_mode = "off"; self.arch = "ens"

    def predict(self, ds, idx, ms=None, noise=None, freeze_gate=False, hidden=False):
        return combine_preds([m.predict(ds, idx, ms, noise, freeze_gate=freeze_gate) for m in self.members])


def split_valid_by_video(ds, frac=None, salt=None):
    """验证集按源视频对半分：视频号加盐做 SHA-256 排序，前 frac 为校准区 VC，其余为选择区 VS。只看视频号，不看标签和模型。"""
    frac = CFG["VALID_CAL_FRAC"] if frac is None else frac; salt = salt or CFG["VALID_SPLIT_SALT"]
    uv = sorted(set(ds.videos), key=lambda v: hashlib.sha256((salt + v).encode()).hexdigest())
    cal_v = set(uv[: int(round(frac * len(uv)))])
    cal = np.array([v in cal_v for v in ds.videos])
    return np.flatnonzero(~cal), np.flatnonzero(cal)


def slice_pred(p, idx, n):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v) for k, v in p.items()}


def slice_panels(panels, idx):
    return {k: [(nm, cb, r, pos, {m: ms[m][idx] for m in MODS}) for nm, cb, r, pos, ms in v] for k, v in panels.items()}


def slice_res(res, idx, n, ds_sub=None):
    """把在整个验证集上得到的评估结果切到子集（VS 或 VC），并在子集上重算指标。"""
    out = dict(full=slice_pred(res["full"], idx, n))
    for kind in ("PM", "LH", "SC", "MS"):
        if kind in res:
            out[kind] = [(nm, cb, r, pos, slice_pred(pr, idx, n)) for nm, cb, r, pos, pr in res[kind]]
    if ds_sub is not None and ds_sub.has_label:
        out["complete"] = metrics(ds_sub.c, ds_sub.y, out["full"]["probs"], out["full"]["yhat"])
        for kind in ("PM", "LH", "SC", "MS"):
            if kind in out:
                df = panel_table(ds_sub, out[kind]); out[kind + "_df"] = df
                out[kind + "_sum"] = panel_summary(df, out["complete"])
    return out


def video_boot(ds, stat_fn, reps=None, seed=0):
    """按源视频有放回重抽样：stat_fn(w) → 标量或数组（w 为样本权重）。返回点估计与 95% 区间。"""
    reps = reps or CFG["BOOT_REPS"]
    rng = np.random.default_rng(seed)
    uv, inv = np.unique(np.array(ds.videos), return_inverse=True)
    pt = np.asarray(stat_fn(np.ones(ds.n)), np.float64)
    bs = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(uv), len(uv)), minlength=len(uv))[inv].astype(np.float64)
        if w.sum() > 0:
            bs.append(np.asarray(stat_fn(w), np.float64))
    bs = np.array(bs)
    return pt, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def weighted_metrics(c_true, y_true, c_pred, y_pred, w):
    C = conf_mat(c_true, c_pred, w)
    return np.array([f1_from_cm(C)[0], np.trace(C) / max(C.sum(), 1e-12), np.sum(w * np.abs(y_pred - y_true)) / max(w.sum(), 1e-12)])

# %% [model] 4. 模型：LoRA 文本编码 + 掩码池化双头 + 缺失门控 LoRA（MG-LoRA）
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, AutoModel, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
log("设备：", DEVICE, "| torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0) if USE_AMP else "-")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else contextlib.nullcontext()


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def load_tokenizer(path):
    return BertTokenizerFast.from_pretrained(path)


class GateHolder:
    """一次前向里暂存每个样本的门控值 g（形状 (B,)），供各层的缺失 LoRA 读取。"""
    def __init__(self):
        self.g = None


class GatedLoRALinear(nn.Module):
    """y = W0 x + s_task·B A x + g·s_miss·B' A' x。
    任务 LoRA（阶段一训练）与缺失 LoRA（阶段二训练）共用一层；g=0 时本层与阶段一逐位相同。"""

    def __init__(self, base, r_task=0, alpha_task=1.0, r_miss=0, alpha_miss=1.0, dropout=0.0, holder=None):
        super().__init__()
        self.base = base
        self.r_task, self.r_miss = r_task, r_miss
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.holder = holder
        fin, fout = base.in_features, base.out_features
        if r_task > 0:
            self.task_A = nn.Parameter(torch.empty(r_task, fin)); nn.init.kaiming_uniform_(self.task_A, a=math.sqrt(5))
            self.task_B = nn.Parameter(torch.zeros(fout, r_task)); self.s_task = alpha_task / r_task
        if r_miss > 0:
            self.miss_A = nn.Parameter(torch.empty(r_miss, fin)); nn.init.kaiming_uniform_(self.miss_A, a=math.sqrt(5))
            self.miss_B = nn.Parameter(torch.zeros(fout, r_miss)); self.s_miss = alpha_miss / r_miss

    def forward(self, x):
        y = self.base(x)
        if self.r_task > 0:
            y = y + F.linear(F.linear(self.drop(x), self.task_A), self.task_B) * self.s_task
        g = self.holder.g if self.holder is not None else None
        if self.r_miss > 0 and g is not None:
            g = g.to(y.dtype).view(-1, *([1] * (x.dim() - 1)))
            y = y + F.linear(F.linear(self.drop(x), self.miss_A), self.miss_B) * (self.s_miss * g)
        return y


class MonoGate(nn.Module):
    """v2 门控：g(φ) = 1 − exp(−aᵀφ)，a = softplus(w) ≥ 0。
    φ 只含连续量（各模态缺失比例、最长缺口比例），于是 g(0)=0、处处连续、对每个缺失量单调不减，
    且 g ≤ aᵀφ ≤ ‖a‖·‖φ‖——补偿强度被缺失严重程度线性地压住。"""

    def __init__(self, d, a0=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.full((d,), math.log(math.expm1(a0))))

    def coef(self):
        return F.softplus(self.w)

    def forward(self, phi):
        return 1.0 - torch.exp(-(phi * self.coef()).sum(-1))


def masked_mean_std(x, a):
    """在可用位置上求均值与标准差；可用位置为空时两者都取 0。"""
    x = x.float(); a = a.float().unsqueeze(-1)
    cnt = a.sum(1)
    has = (cnt > 0).float()
    c = cnt.clamp(min=1.0)
    mu = (x * a).sum(1) / c
    var = (((x - mu.unsqueeze(1)) ** 2) * a).sum(1) / c
    sd = torch.sqrt(var + 1e-8) * has
    return torch.cat([mu * has, sd], dim=-1)


class MGModel(nn.Module):
    def __init__(self, dA, dV):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH, add_pooling_layer=False)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.hT, self.hF = GateHolder(), GateHolder()
        for layer in self.bert.encoder.layer:
            att = layer.attention.self
            for nm in ("query", "value"):
                setattr(att, nm, GatedLoRALinear(getattr(att, nm), CFG["LORA_R"], CFG["LORA_ALPHA"],
                                                 CFG["MISS_R_TEXT"], CFG["MISS_ALPHA"], CFG["LORA_DROPOUT"], self.hT))
        Hd = self.bert.config.hidden_size
        din = 2 * (Hd + dA + dV)
        rF, aF = CFG["MISS_R_FUSE"], CFG["MISS_ALPHA"]
        self.fc1 = GatedLoRALinear(nn.Linear(din, CFG["HIDDEN"]), 0, 1, rF, aF, 0.0, self.hF)
        self.fc2 = GatedLoRALinear(nn.Linear(CFG["HIDDEN"], CFG["FUSE_DIM"]), 0, 1, rF, aF, 0.0, self.hF)
        self.head_c = GatedLoRALinear(nn.Linear(CFG["FUSE_DIM"], 3), 0, 1, 4, aF, 0.0, self.hF)
        self.head_r = GatedLoRALinear(nn.Linear(CFG["FUSE_DIM"], 1), 0, 1, 4, aF, 0.0, self.hF)
        self.drop = nn.Dropout(CFG["DROPOUT"])
        self.gate_T = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.gate_F = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 1))
        self.gate_T2 = MonoGate(2, CFG["MONO_INIT"])     # φ_T 的连续部分：文本缺失比例、最长缺口
        self.gate_F2 = MonoGate(6, CFG["MONO_INIT"])     # 三个模态的缺失比例与最长缺口
        self.gate_mode = "off"      # off：阶段一/基础模型；gated：MG-LoRA v1；mono：v2 单调门控；always：无门控消融

    arch = "bert"
    ordinal = False

    def make_batch(self, ds, idx, ms=None, noise=None):
        return to_batch(ds, idx, ms, noise)

    def encode_rows(self, b, ri=None):
        if ri is None:
            return self.encode_text(b["ids"], b["attn"], b["tt"], b["aT"], b["phiT"], b["indT"])
        return self.encode_text(b["ids"][ri], b["attn"][ri], b["tt"][ri], b["aT"][ri], b["phiT"][ri], b["indT"][ri])

    def gate(self, net, phi, ind):
        if self.gate_mode == "off":
            return None
        if self.gate_mode == "always":
            return torch.ones(phi.shape[0], device=phi.device)
        if self.gate_mode == "mono":
            if phi.shape[1] == 3:
                return self.gate_T2(phi[:, :2])
            return self.gate_F2(phi[:, [0, 1, 3, 4, 6, 7]])
        return torch.sigmoid(net(phi).squeeze(-1)) * ind

    def encode_text(self, ids, attn, tt, aT, phiT, indT):
        self.hT.g = self.gate(self.gate_T, phiT, indT)
        try:
            with autocast():
                hs = self.bert(input_ids=ids, attention_mask=attn, token_type_ids=tt).last_hidden_state
        finally:
            self.hT.g = None
        return masked_mean_std(hs, aT)

    def fuse(self, zT, audio, vision, aA, aV, phi, indF):
        z = torch.cat([zT, masked_mean_std(audio, aA), masked_mean_std(vision, aV)], dim=-1)
        g = self.gate(self.gate_F, phi, indF)
        self.hF.g = g
        try:
            with autocast():
                h = self.drop(F.relu(self.fc1(z)))
                h = self.drop(F.relu(self.fc2(h)))
                logits = self.head_c(h)
                u = self.head_r(h).squeeze(-1)
        finally:
            self.hF.g = None
        return logits.float(), 3 * torch.tanh(u.float()), h.float(), g

    def forward_batch(self, b):
        zT = self.encode_text(b["ids"], b["attn"], b["tt"], b["aT"], b["phiT"], b["indT"])
        return self.fuse(zT, b["audio"], b["vision"], b["aA"], b["aV"], b["phi"], b["indF"])


def _t(x, dt=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dt, device=DEVICE)


def to_batch(ds, idx, ms=None, noise=None):
    """ms：与 idx 行对齐的删除掩码 {'T','A','V'}；noise：(rng, sd) 给声学、视觉可用位置加噪声（稳定性检验用）。"""
    idx = np.asarray(idx)
    aT, aA, aV, attn = apply_masks(ds, idx, ms)
    phi, indT, indF = descriptors(aT, aA, aV, ref_rows(ds, idx))
    audio, vision = ds.audio[idx], ds.vision[idx]
    if noise is not None:
        rng, sd = noise
        audio = audio + rng.normal(0, sd, audio.shape).astype(np.float32) * aA[..., None]
        vision = vision + rng.normal(0, sd, vision.shape).astype(np.float32) * aV[..., None]
    return dict(ids=_t(ds.input_ids[idx], torch.long), attn=_t(attn, torch.long), tt=_t(ds.token_type[idx], torch.long),
                aT=_t(aT), aA=_t(aA), aV=_t(aV), audio=_t(audio), vision=_t(vision),
                phi=_t(phi), phiT=_t(phi[:, :3]), indT=_t(indT), indF=_t(indF),
                emptyT=aT.sum(1) == 0, emptyAll=(aT.sum(1) + aA.sum(1) + aV.sum(1)) == 0)


OPTIONAL_KEYS = ("gate_T2.", "gate_F2.")      # v1 检查点里没有 v2 门控参数，读取时允许缺失


FROZEN_PREFIX = ("bert.", "enc.")                # 冻结的预训练主干（BERT / DeBERTa），检查点里不存


def _is_trainable_key(k):
    return (not k.startswith(FROZEN_PREFIX)) or ("task_" in k) or ("miss_" in k)


def load_trainable(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if _is_trainable_key(k) and not k.startswith(OPTIONAL_KEYS)]
    if unexpected or bad:
        raise RuntimeError(f"检查点与模型不匹配：unexpected={unexpected[:5]} missing={bad[:5]}")


class Shared:
    """同一种主干的所有配置共用一个 GPU 上的模型实例：冻结主干完全相同，切换配置只换 LoRA、头部与门控参数。
    v3 有两种主干（bert / deb），各一个实例。"""
    models = {}
    keys = {}
    dims = None


class TorchPredictor:
    """统一预测接口：predict(ds, idx, ms=None, noise=None) → dict(probs, yhat, gT, gF)。
    完整文本的编码按数据集缓存；只有文本被删改的行才重新过 BERT，文本整段缺失的行直接取 0。"""

    def __init__(self, state=None, gate_mode="off", key=None, model=None, prior=None, arch="bert"):
        self.state, self.gate_mode, self.key, self.own = state, gate_mode, key, model
        self.arch = model.arch if model is not None else arch
        pp, py = prior if prior is not None else PRIOR
        self.prior_probs = np.asarray(pp, np.float32); self.prior_y = float(py)
        self._cache = {}

    def _model(self):
        if self.own is not None:
            return self.own
        a = self.arch
        if Shared.models.get(a) is None:
            Shared.models[a] = (DebModel if a == "deb" else MGModel)(*Shared.dims).to(DEVICE)
        m = Shared.models[a]
        if Shared.keys.get(a) != self.key:
            load_trainable(m, self.state)
            Shared.keys[a] = self.key
        m.gate_mode = self.gate_mode
        return m

    @torch.no_grad()
    def _text_full(self, ds, model):
        if ds.uid not in self._cache:
            zs = []
            bs_ = CFG["DEB_EVAL_BS"] if model.arch == "deb" else CFG["EVAL_BS"]
            for st in range(0, ds.n, bs_):
                idx = np.arange(st, min(ds.n, st + bs_))
                b = model.make_batch(ds, idx)
                z = model.encode_rows(b)
                z[_t(np.flatnonzero(b["emptyT"]), torch.long)] = 0
                zs.append(z)
            self._cache[ds.uid] = torch.cat(zs) if zs else None
        return self._cache[ds.uid]

    @torch.no_grad()
    def predict(self, ds, idx, ms=None, noise=None, freeze_gate=False, hidden=False):
        """freeze_gate=True：门控按“未遮挡时”的缺失状态计算（局部遮挡只探测内容贡献，不触发缺失补偿）。
        hidden=True：另外返回文本池化表示 zT 与融合隐层 h（低秩检验、表示偏差分析用）。"""
        model = self._model()
        model.eval()
        idx = np.asarray(idx, dtype=np.int64)
        n = len(idx)
        out = dict(probs=np.zeros((n, 3), np.float32), yhat=np.zeros(n, np.float32),
                   gT=np.zeros(n, np.float32), gF=np.zeros(n, np.float32))
        if model.ordinal:
            out["dist"] = np.zeros((n, CFG["NGRID"]), np.float32)
        if hidden:
            out["zT"] = []; out["h"] = []
        if n == 0:
            return out
        zfull = self._text_full(ds, model)
        ebs = CFG["DEB_EVAL_BS"] if model.arch == "deb" else CFG["EVAL_BS"]
        for st in range(0, n, ebs):
            sl = slice(st, min(n, st + ebs))
            bidx = idx[sl]
            bms = None if ms is None else {m: ms[m][sl] for m in MODS}
            b = model.make_batch(ds, bidx, bms, noise)
            if freeze_gate and bms is not None:
                aT0, aA0, aV0, _ = apply_masks(ds, bidx, None)
                phi0, indT0, indF0 = descriptors(aT0, aA0, aV0, ref_rows(ds, bidx))
                b.update(phi=_t(phi0), phiT=_t(phi0[:, :3]), indT=_t(indT0), indF=_t(indF0))
            zT = zfull[_t(bidx, torch.long)].clone()
            if bms is not None:
                changed = bms["T"].any(1)
                empty = b["emptyT"]
                rows = np.flatnonzero(changed & ~empty)
                if len(rows):
                    ri = _t(rows, torch.long)
                    zT[ri] = model.encode_rows(b, ri)
                if empty.any():
                    zT[_t(np.flatnonzero(empty), torch.long)] = 0
            gT = model.gate(model.gate_T, b["phiT"], b["indT"])
            logits, yhat, hF, gF = model.fuse(zT, b["audio"], b["vision"], b["aA"], b["aV"], b["phi"], b["indF"])
            if hidden:
                out["zT"].append(zT.float().cpu().numpy()); out["h"].append(hF.cpu().numpy())
            p = torch.softmax(logits, -1).cpu().numpy()
            y = yhat.cpu().numpy()
            if model.ordinal:
                dd = torch.softmax(model.last_ord, -1).cpu().numpy()
                y = dist_median(dd)
            ea = b["emptyAll"]
            if ea.any():
                p[ea] = self.prior_probs; y[ea] = self.prior_y
                if model.ordinal:
                    dd[ea] = self.prior_dist()
            out["probs"][sl] = p; out["yhat"][sl] = y
            if model.ordinal:
                out["dist"][sl] = dd
            if gT is not None: out["gT"][sl] = gT.float().cpu().numpy()
            if gF is not None: out["gF"][sl] = gF.float().cpu().numpy()
        if hidden:
            out["zT"] = np.concatenate(out["zT"]); out["h"] = np.concatenate(out["h"])
        return out

    def prior_dist(self):
        """全部模态缺失时的兜底强度分布：训练标签落在最近格点上的频率。"""
        if not hasattr(self, "_pd"):
            y = data["train"].y if "data" in globals() else np.array([self.prior_y])
            k = np.clip(np.rint((y + 3) * 3).astype(int), 0, CFG["NGRID"] - 1)
            self._pd = (np.bincount(k, minlength=CFG["NGRID"]) / max(1, len(k))).astype(np.float32)
        return self._pd

    @torch.no_grad()
    def token_states(self, ds, idx):
        """完整输入下 BERT 最后一层的逐位置表示（命题 3 的自相关估计用）。"""
        model = self._model(); model.eval()
        hs = []
        for st in range(0, len(idx), CFG["EVAL_BS"]):
            b = to_batch(ds, np.asarray(idx[st:st + CFG["EVAL_BS"]]))
            model.hT.g = model.gate(model.gate_T, b["phiT"], b["indT"])
            try:
                with autocast():
                    hs.append(model.bert(input_ids=b["ids"], attention_mask=b["attn"],
                                         token_type_ids=b["tt"]).last_hidden_state.float().cpu().numpy())
            finally:
                model.hT.g = None
        return np.concatenate(hs)

    def gate_coef(self):
        m = self._model()
        return dict(text=m.gate_T2.coef().detach().cpu().numpy().tolist(),
                    fusion=m.gate_F2.coef().detach().cpu().numpy().tolist())


# %% [train] 5. 训练：阶段一（base / joint）与阶段二（mg / mg_nogate / mg_nokd / mg_scatter）
def class_weights(tr):
    cnt = np.bincount(tr.c, minlength=3).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1)) if CFG["CLASS_WEIGHT"] == "sqrt_inv" else np.ones(3)
    return _t(w / w.mean())


def lr_lambda(total, warm):
    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return max(0.0, (total - step) / max(1, total - warm))
    return f


def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if _is_trainable_key(k)}


def param_counts(model):
    cnt = lambda f: int(sum(p.numel() for n, p in model.named_parameters() if f(n)))
    return dict(total=cnt(lambda n: True), bert_frozen=cnt(lambda n: n.startswith("bert.") and "task_" not in n and "miss_" not in n),
                stage1_task_lora=cnt(lambda n: "task_" in n),
                stage1_heads=cnt(lambda n: n.startswith(("fc1.base", "fc2.base", "head_c.base", "head_r.base"))),
                stage2_miss_lora_text=cnt(lambda n: "miss_" in n and n.startswith("bert.")),
                stage2_miss_lora_fuse=cnt(lambda n: "miss_" in n and not n.startswith("bert.")),
                stage2_gates=cnt(lambda n: n.startswith(("gate_T", "gate_F"))))


def quick_eval(model, data, with_panel):
    pr = TorchPredictor(model=model)
    va = data["valid"]
    full = pr.predict(va, np.arange(va.n))
    m = metrics(va.c, va.y, full["probs"], full["yhat"])
    res = dict(f1=m["f1"], mae=m["mae"], acc=m["acc"])
    if with_panel:
        preds = [(nm, cb, r, pos, pr.predict(va, np.arange(va.n), ms)) for nm, cb, r, pos, ms in PANELS["valid"]["PM"]]
        df = panel_table(va, preds)
        res.update(pm_f1=float(df.f1.mean()), pm_mae=float(df.mae.mean()), pm_worst=float(df.f1.min()))
    return res


def _step(opt, sch, scaler, loss, params):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); sch.step()


def train_stage1(data, seed, mode="base"):
    """mode=base：只用完整输入；mode=joint：常规做法，把缺失样本混进来一起训练（对照）。"""
    set_seed(seed)
    tr = data["train"]
    model = MGModel(tr.audio.shape[2], tr.vision.shape[2]).to(DEVICE)
    model.gate_mode = "off"
    task = [p for n, p in model.named_parameters() if "task_" in n]
    head = [p for n, p in model.named_parameters() if n.startswith(("fc1.base", "fc2.base", "head_c.base", "head_r.base"))]
    for p in task + head:
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": task, "lr": CFG["S1_LR_LORA"]}, {"params": head, "lr": CFG["S1_LR_HEAD"]}],
                            weight_decay=CFG["S1_WD"])
    bs = CFG["S1_BS"]; steps = CFG["S1_EPOCHS"] * math.ceil(tr.n / bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(0.1 * steps)))
    scaler = make_scaler(); cw = class_weights(tr); rng = np.random.default_rng(seed)
    best_key, best_state, hist = None, None, []
    for ep in range(CFG["S1_EPOCHS"]):
        model.train(); perm = rng.permutation(tr.n); tl = []
        for st in range(0, tr.n, bs):
            idx = perm[st:st + bs]
            ms = sample_family_masks(tr, idx, rng, p_masked=CFG["JOINT_MASK_P"])[0] if mode == "joint" else None
            logits, yhat, _, _ = model.forward_batch(to_batch(tr, idx, ms))
            loss = F.cross_entropy(logits, _t(tr.c[idx], torch.long), weight=cw) + F.l1_loss(yhat, _t(tr.y[idx]))
            _step(opt, sch, scaler, loss, task + head)
            tl.append(loss.item())
        ev = quick_eval(model, data, with_panel=(mode == "joint"))
        ev.update(epoch=ep + 1, train_loss=float(np.mean(tl)))
        hist.append(ev)
        key = (ev["pm_f1"], -ev["pm_mae"]) if mode == "joint" else (ev["f1"], -ev["mae"])
        log(f"  [{mode} s{seed}] ep{ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, float)))
        if best_key is None or key > best_key:
            best_key, best_state = key, trainable_state(model)
    load_trainable(model, best_state)
    return model, hist, best_state


def train_stage2(data, seed, base_state, variant="mg"):
    """冻结阶段一的全部参数，只训练缺失 LoRA 与门控网络。
    mg：门控 + 完整视角自蒸馏；mg_nogate：门控恒为 1；mg_nokd：去掉教师（蒸馏与表示一致项）；
    mg_scatter：同 mg，但训练缺失改为散点（检验“成段”这一题面约束是否必要）。"""
    set_seed(seed + 1000)
    tr = data["train"]
    model = MGModel(tr.audio.shape[2], tr.vision.shape[2]).to(DEVICE)
    load_trainable(model, base_state)
    for n, p in model.named_parameters():
        p.requires_grad = ("miss_" in n) or n.startswith(("gate_T", "gate_F"))
    params = [p for p in model.parameters() if p.requires_grad]
    gate_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("gate_T", "gate_F"))]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("gate_T", "gate_F"))]
    opt = torch.optim.AdamW([{"params": other, "weight_decay": 0.01}, {"params": gate_params, "weight_decay": 0.0}],
                            lr=CFG["S2_LR"])
    bs = CFG["S2_BS"]; steps = CFG["S2_EPOCHS"] * math.ceil(tr.n / bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(0.1 * steps)))
    scaler = make_scaler(); cw = class_weights(tr); rng = np.random.default_rng(seed + 1000)
    mode = {"mg_nogate": "always", "mg2": "mono"}.get(variant, "gated")
    p_masked = 1.0 - CFG["NOGATE_COMPLETE_P"] if variant == "mg_nogate" else 1.0
    family = "scatter" if variant == "mg_scatter" else "contig"
    lam_kd = 0.0 if variant == "mg_nokd" else CFG["LAMBDA_KD"]
    lam_ft = 0.0 if variant == "mg_nokd" else CFG["LAMBDA_FEAT"]
    Tk = CFG["KD_T"]
    best_key, best_state, hist = None, None, []
    if variant == "mg2" and CFG["EPOCH0_CANDIDATE"]:
        # epoch 0 = “不补偿”（缺失 LoRA 的 B′ 仍为 0，输出与阶段一逐位相同），也作为一个候选参与选择
        model.gate_mode = mode
        ev0 = quick_eval(model, data, with_panel=True); ev0.update(epoch=0, train_loss=float("nan"))
        hist.append(ev0)
        best_key, best_state = (ev0["pm_f1"], -ev0["pm_mae"]), trainable_state(model)
        log(f"  [{variant} s{seed}] ep0(不补偿): pm_f1={ev0['pm_f1']:.4f}, pm_mae={ev0['pm_mae']:.4f}")
    for ep in range(CFG["S2_EPOCHS"]):
        perm = rng.permutation(tr.n); tl = []
        for st in range(0, tr.n, bs):
            idx = perm[st:st + bs]
            ms, _ = sample_family_masks(tr, idx, rng, p_masked=p_masked, family=family)
            with torch.no_grad():
                model.eval(); model.gate_mode = "off"
                lt, yt, ht, _ = model.forward_batch(to_batch(tr, idx))
            model.train(); model.gate_mode = mode
            ls, ys, hs, _ = model.forward_batch(to_batch(tr, idx, ms))
            loss = F.cross_entropy(ls, _t(tr.c[idx], torch.long), weight=cw) + F.l1_loss(ys, _t(tr.y[idx]))
            if lam_kd > 0:
                kd = F.kl_div(F.log_softmax(ls / Tk, -1), F.softmax(lt / Tk, -1), reduction="batchmean") * Tk * Tk
                loss = loss + lam_kd * (kd + F.l1_loss(ys, yt))
            if lam_ft > 0:
                loss = loss + lam_ft * F.mse_loss(hs, ht)
            _step(opt, sch, scaler, loss, params)
            tl.append(loss.item())
        model.gate_mode = mode
        ev = quick_eval(model, data, with_panel=True)
        ev.update(epoch=ep + 1, train_loss=float(np.mean(tl)))
        hist.append(ev)
        log(f"  [{variant} s{seed}] ep{ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, float)))
        key = (ev["pm_f1"], -ev["pm_mae"])
        if best_key is None or key > best_key:
            best_key, best_state = key, trainable_state(model)
    load_trainable(model, best_state)
    model.gate_mode = mode
    sel = max(hist, key=lambda h: (h["pm_f1"], -h["pm_mae"]))["epoch"]
    for h in hist:
        h["selected"] = int(h["epoch"] == sel)
    log(f"  [{variant} s{seed}] 选中 epoch {sel}")
    return model, hist, best_state


GATE_MODE = dict(base="off", joint="off", mg="gated", mg_nokd="gated", mg_scatter="gated", mg_nogate="always", mg2="mono")
TRAIN_KEYS = ["S1_EPOCHS", "S1_BS", "S1_LR_LORA", "S1_LR_HEAD", "S1_WD", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT",
              "HIDDEN", "FUSE_DIM", "DROPOUT", "CLASS_WEIGHT", "S2_EPOCHS", "S2_BS", "S2_LR", "MISS_R_TEXT",
              "MISS_R_FUSE", "MISS_ALPHA", "KD_T", "LAMBDA_KD", "LAMBDA_FEAT", "NOGATE_COMPLETE_P", "JOINT_MASK_P",
              "FAM_WHOLE_P", "FAM_R_MIN", "FAM_R_MAX", "FAM_MAX_SEG", "PANEL_RATIOS", "SMOKE"]


CKPT_NAME = re.compile(r"^(base|joint|mg|mg2|mg_nogate|mg_nokd|mg_scatter)_s\d+\.pt$")
CKPT_NAME_DEB = re.compile(r"^deb_s\d+\.pt$")
DEB_KEYS = ["TEXT_MODEL", "DEB_MAXLEN", "DEB_BS", "DEB_EPOCHS", "DEB_LR_LORA", "DEB_LR_HEAD", "DEB_WD", "DEB_WARMUP",
            "LORA3_R", "LORA3_ALPHA", "LORA3_DROPOUT", "LORA3_TARGETS", "NGRID", "ORD_SIGMA", "W_ORD", "W_L1", "W_SF1",
            "RDROP_ALPHA", "AUG_P", "HIDDEN", "FUSE_DIM", "DROPOUT", "CLASS_WEIGHT", "FAM_WHOLE_P", "FAM_R_MIN",
            "FAM_R_MAX", "FAM_MAX_SEG", "VALID_SPLIT_SALT", "VALID_CAL_FRAC", "SMOKE"]


def ckpt_dir(data, keys=None, prefix="", name_re=None, cfg_name="train_config.json"):
    """检查点目录 OUT/ckpt/<前缀><签名>。再到 CKPT_SEARCH 下找已有的 .pt 复制过来：
    ① 在同签名文件夹里的直接用；② 旁边有配置文件的先核对配置，不一致就跳过（防止误用冒烟模型）；
    ③ 平铺上传、没有任何配置信息的也用，但会给出提醒。"""
    keys = keys or TRAIN_KEYS; name_re = name_re or CKPT_NAME
    sig_src = {k: CFG[k] for k in keys}
    sig_src["n_train"] = data["train"].n; sig_src["n_valid"] = data["valid"].n
    sig = prefix + hashlib.md5(json.dumps(sig_src, sort_keys=True, default=str).encode()).hexdigest()[:8]
    d = OUT / "ckpt" / sig
    d.mkdir(parents=True, exist_ok=True)
    found = []
    for root in CFG["CKPT_SEARCH"]:
        for p in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True):
            if str(p).startswith(str(OUT)) or not name_re.match(os.path.basename(p)):
                continue
            parent = Path(p).parent; cj = parent / cfg_name
            if parent.name == sig:
                prio = 0
            elif cj.exists():
                try:
                    other = json.load(open(cj))
                    same = all(json.dumps(other.get(k), default=str) == json.dumps(v, default=str) for k, v in sig_src.items())
                except Exception:
                    same = False
                if not same:
                    log("  跳过（训练配置不一致，可能是冒烟模型）：", p); continue
                prio = 1
            else:
                prio = 2
            found.append((prio, p))
    warned = False
    for prio, p in sorted(found):
        q = d / os.path.basename(p)
        if q.exists():
            continue
        shutil.copy(p, q)
        log(f"  复用检查点{'（未附训练配置，默认它与当前配置一致）' if prio == 2 else ''}：", p)
        warned |= prio == 2
    if warned:
        log(f"  提醒：有检查点没有附 {cfg_name}，无法自动核对；后面会再用验证集成绩做一次合理性检查。")
    save_json(sig_src, d / cfg_name)
    return d


def _load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def v3_bert_plan(ck_dir):
    """v3 的 BERT 系模型：base / mg 5 个种子；joint 补到 5 个种子（对照）；mg2 有检查点才读（或按开关重训）。"""
    plan = [("base", s) for s in CFG["SEEDS_MAIN"]] + [("mg", s) for s in CFG["SEEDS_MAIN"]]
    plan += [("joint", s) for s in (CFG["SEEDS_MAIN"] if CFG["JOINT_FILL_SEEDS"] else CFG["SEEDS_ABL"])]
    if CFG["AUDIT_RETRAIN_MG2"]:
        plan += [("mg2", s) for s in CFG["SEEDS_MAIN"]]
    else:
        miss = [s for s in CFG["SEEDS_MAIN"] if not (ck_dir / f"mg2_s{s}.pt").exists()]
        plan += [("mg2", s) for s in CFG["SEEDS_MAIN"] if s not in miss]
        if miss:
            log(f"  mg2 缺检查点的种子 {miss} 不参与本次评估（AUDIT_RETRAIN_MG2=False）")
    return plan


def run_training(data, plan_fn=None):
    """训练（或从检查点续跑）全部配置，返回 {(config, seed): TorchPredictor} 与训练曲线。"""
    tr = data["train"]; Shared.dims = (tr.audio.shape[2], tr.vision.shape[2])
    ck_dir = ckpt_dir(data)
    if plan_fn is not None:
        plan = plan_fn(ck_dir)
    else:
        plan = [("base", s) for s in CFG["SEEDS_MAIN"]] + [("mg", s) for s in CFG["SEEDS_MAIN"]]
        plan += [("joint", s) for s in CFG["SEEDS_ABL"]]
        plan += [("mg2", s) for s in CFG["SEEDS_MAIN"]]
        if CFG["RUN_ABLATIONS"]:
            plan += [(v, s) for v in CFG["ABLATIONS"] if v != "joint" for s in CFG["SEEDS_ABL"]]
    have = [p for p in plan if (ck_dir / f"{p[0]}_s{p[1]}.pt").exists()]
    log(f"检查点：{len(have)}/{len(plan)} 个已存在；需要新训：{[f'{c}_s{s}' for c, s in plan if (c, s) not in have]}")
    if not any(p[0] == "base" for p in have) and not CFG["SMOKE"]:
        log("  注意：没有找到 v1 的检查点，base/mg/joint 会重新训练（约 +1.5 小时）。"
            "如需复用，请把 v1 那次 Version 的输出 Add Input 进来。")
    preds, hists, timing = {}, {}, {}
    for cfg_name, seed in plan:
        ck = ck_dir / f"{cfg_name}_s{seed}.pt"
        if ck.exists():
            log(f"读取已有检查点 {ck.name}")
            state = _load_state(ck)
        else:
            if not CFG["RUN_TRAIN"]:
                raise FileNotFoundError(f"缺少检查点 {ck}，且 RUN_TRAIN=False")
            log(f"==== 训练 {cfg_name} seed {seed} ====")
            t1 = time.time()
            if cfg_name in ("base", "joint"):
                model, hist, state = train_stage1(data, seed, mode=cfg_name)
            else:
                base_ck = ck_dir / f"base_s{seed}.pt"
                if not base_ck.exists():
                    m0, h0, st0 = train_stage1(data, seed, "base"); torch.save(st0, base_ck)
                    pd.DataFrame(h0).to_csv(OUT / "runs" / f"history_base_s{seed}.csv", index=False)
                    del m0
                model, hist, state = train_stage2(data, seed, _load_state(base_ck), cfg_name)
            torch.save(state, ck)
            hists[(cfg_name, seed)] = hist
            timing[f"{cfg_name}_s{seed}"] = time.time() - t1
            pd.DataFrame(hist).to_csv(OUT / "runs" / f"history_{cfg_name}_s{seed}.csv", index=False)
            log(f"  用时 {time.time() - t1:.0f}s")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        preds[(cfg_name, seed)] = TorchPredictor(state, GATE_MODE[cfg_name], key=(cfg_name, seed))
    save_json(timing, OUT / "tables" / "train_time_seconds.json")
    try:
        pc = param_counts(preds[("mg", CFG["DEPLOY_SEED"])]._model())
        save_json(pc, OUT / "tables" / "param_counts.json"); log("参数量：", pc)
    except Exception as e:
        log("参数量统计失败：", e)
    S3.setdefault("bert_plan", [f"{c}_s{s}" for c, s in plan])
    S3.setdefault("bert_trained_now", list(timing.keys()))
    return preds, hists


# %% [deb] 5b. v3 · DeBERTa-v3-large 主干：BERT 子词位置 → 原文字符区间 → DeBERTa 词元（逐位缺失映射）；模型；训练
import unicodedata
from collections import defaultdict

_NOSPACE_BEFORE = set(".,!?;:%)]}")
_CONTRACT = {"s", "t", "re", "ve", "ll", "d", "m"}
DEB_ROW_ATTRS = ("deb_ids", "deb_attn", "deb_real", "deb_map", "deb_len")


def load_deb_tokenizer(path):
    """优先用 fast 分词器（需要字符偏移）；转换失败时退回慢速分词器（逐词切分）。"""
    try:
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        if getattr(tok, "is_fast", False):
            return tok
    except Exception as e:
        log("  DeBERTa fast 分词器加载失败，改用慢速：", repr(e)[:160])
    return AutoTokenizer.from_pretrained(path, use_fast=False)


def _bert_pieces(ds, i, btok):
    """词范围内每个位置：(位置, 子词串（去掉 ##）, 是否续接片段)。"""
    toks = btok.convert_ids_to_tokens([int(x) for x in ds.input_ids[i]])
    return [(int(t), toks[t][2:] if toks[t].startswith("##") else toks[t], toks[t].startswith("##"))
            for t in np.flatnonzero(ds.W[i])]


def _norm_chars(raw):
    """原文 → 规范化字符（与 bert-base-uncased 相同：NFD 去重音 + 小写）及其在原文中的下标。"""
    chars, where = [], []
    for j, ch in enumerate(raw):
        for c in unicodedata.normalize("NFD", ch):
            if unicodedata.category(c) == "Mn":
                continue
            for c2 in c.lower():
                chars.append(c2); where.append(j)
    return chars, where


def _align_raw(pieces, raw):
    """把 BERT 子词依次对到原文字符上。成功：返回（原文截到最后一个子词, {位置: 原文字符区间}）；对不上返回 None。"""
    if not raw or not pieces or any(p in ("[UNK]", "[PAD]") for _, p, _ in pieces):
        return None                        # 有 [PAD]（原生缺失的文本位置）时不用原文，以免 DeBERTa 看到已缺失的词
    chars, where = _norm_chars(raw)
    k, spans = 0, {}
    for t, p, _ in pieces:
        while k < len(chars) and chars[k].isspace():
            k += 1
        if not p or "".join(chars[k:k + len(p)]) != p:
            return None
        spans[t] = (where[k], where[k + len(p) - 1] + 1)
        k += len(p)
    end = max(e for _, e in spans.values())
    return raw[:end], spans


def _detok(pieces):
    """原文对不上时的退路：用 BERT 子词拼回一句（小写），同时记录每个位置的字符区间。"""
    txt, spans, prev = "", {}, None
    for t, p, cont in pieces:
        if p == "[PAD]":                   # 原生缺失的文本位置：不产生任何文字，也就没有对应的 DeBERTa 词元
            continue
        glue = cont or not txt or (p[:1] in _NOSPACE_BEFORE) or p == "'" or (prev == "'" and p in _CONTRACT)
        if not glue:
            txt += " "
        spans[t] = (len(txt), len(txt) + len(p)); txt += p
        prev = p
    return txt, spans


def prepare_deb(ds, btok, dtok):
    """给一个数据集准备 DeBERTa 输入：词元号、注意力、是否实词元、以及“DeBERTa 词元 × BERT 位置”的重叠矩阵。
    推理/训练时：某个 BERT 位置不可用 → 与它字符重叠的 DeBERTa 词元全部视为缺失（保守，不泄漏被删内容）。"""
    L, n, T = CFG["DEB_MAXLEN"], ds.n, ds.T
    pad = dtok.pad_token_id if dtok.pad_token_id is not None else 0
    ids = np.full((n, L), pad, np.int64); attn = np.zeros((n, L), np.int8)
    real = np.zeros((n, L), bool); mp = np.zeros((n, L, T), bool); ln = np.zeros(n, np.int64)
    st = dict(raw_aligned=0, detok=0, truncated=0, empty_map_tokens=0, fast=bool(getattr(dtok, "is_fast", False)))
    special = set(dtok.all_special_ids)
    for i in range(n):
        pieces = _bert_pieces(ds, i, btok)
        al = _align_raw(pieces, ds.raw_text[i])
        if al is not None:
            txt, spans = al; st["raw_aligned"] += 1
        else:
            txt, spans = _detok(pieces); st["detok"] += 1
        pos = np.array(sorted(spans)); S0 = np.array([spans[t][0] for t in pos]); S1 = np.array([spans[t][1] for t in pos])
        if st["fast"]:
            enc = dtok(txt, add_special_tokens=True, truncation=True, max_length=L, return_offsets_mapping=True)
            tid, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
            rows = []
            for j, (a, b) in enumerate(offs):
                if tid[j] in special and a == b:
                    rows.append(None); continue
                hit = pos[(S0 < b) & (a < S1)] if len(pos) else np.array([], int)
                rows.append(hit)
            covered_end = max([b for (a, b), r in zip(offs, rows) if r is not None] + [0])
            if len(pos) and covered_end < S1.max():
                st["truncated"] += 1
        else:                                                   # 慢速分词器：按 BERT 整词逐个切
            words, cur = [], []
            for t, p, cont in pieces:
                if p == "[PAD]":
                    continue
                if cont and cur:
                    cur.append((t, p))
                else:
                    if cur: words.append(cur)
                    cur = [(t, p)]
            if cur: words.append(cur)
            tid, rows, trunc = [dtok.cls_token_id], [None], False
            for w in words:
                wt = dtok.convert_tokens_to_ids(dtok.tokenize("".join(p for _, p in w)))
                for x in wt:
                    if len(tid) >= L - 1:
                        trunc = True; break
                    tid.append(x); rows.append(np.array([t for t, _ in w]))
                if trunc:
                    break
            st["truncated"] += int(trunc)
            tid.append(dtok.sep_token_id); rows.append(None)
        # 只覆盖空白的词元（如单独的 “▁”）：借用后一个（没有则前一个）实词元的映射
        for j in range(len(rows)):
            if rows[j] is not None and len(rows[j]) == 0:
                st["empty_map_tokens"] += 1
                nxt = next((rows[k] for k in range(j + 1, len(rows)) if rows[k] is not None and len(rows[k])), None)
                prv = next((rows[k] for k in range(j - 1, -1, -1) if rows[k] is not None and len(rows[k])), None)
                rows[j] = nxt if nxt is not None else (prv if prv is not None else np.array([], int))
        m = len(tid)
        ids[i, :m] = tid; attn[i, :m] = 1; ln[i] = m
        for j, r in enumerate(rows):
            if r is not None:
                real[i, j] = True
                if len(r):
                    mp[i, j, r] = True
    ds.deb_ids, ds.deb_attn, ds.deb_real, ds.deb_map, ds.deb_len = ids, attn, real, mp, ln
    ds.row_extra = DEB_ROW_ATTRS
    st.update(n=n, mean_len=float(ln.mean()) if n else 0.0, max_len=int(ln.max()) if n else 0,
              bert_pos_without_token=int((ds.W & ~mp.any(1)).sum()))
    return st


def to_batch_deb(ds, idx, ms=None, noise=None):
    idx = np.asarray(idx)
    aT, aA, aV, _ = apply_masks(ds, idx, ms)
    phi, indT, indF = descriptors(aT, aA, aV, ref_rows(ds, idx))
    miss_b = ds.W[idx] & ~aT                                   # 词范围内不可用的 BERT 位置（原生 + 人为删除）
    real = ds.deb_real[idx]
    tok_miss = (ds.deb_map[idx] & miss_b[:, None, :]).any(-1)
    aTd = real & ~tok_miss
    attn = ds.deb_attn[idx].astype(np.int64)
    attn[real & tok_miss] = 0
    Lb = int(max(2, ds.deb_len[idx].max()))
    audio, vision = ds.audio[idx], ds.vision[idx]
    if noise is not None:
        rng, sd = noise
        audio = audio + rng.normal(0, sd, audio.shape).astype(np.float32) * aA[..., None]
        vision = vision + rng.normal(0, sd, vision.shape).astype(np.float32) * aV[..., None]
    return dict(ids=_t(ds.deb_ids[idx][:, :Lb], torch.long), attn=_t(attn[:, :Lb], torch.long), tt=None,
                aT=_t(aTd[:, :Lb]), aA=_t(aA), aV=_t(aV), audio=_t(audio), vision=_t(vision),
                phi=_t(phi), phiT=_t(phi[:, :3]), indT=_t(indT), indF=_t(indF),
                emptyT=aTd.sum(1) == 0, emptyAll=(aTd.sum(1) + aA.sum(1) + aV.sum(1)) == 0)


class DebModel(nn.Module):
    """DeBERTa-v3-large（冻结）+ 全部线性层 LoRA + 掩码均值/标准差池化 + 融合 MLP；
    两个头：三分类 logits；19 点有序强度分布（点估计取分布中位数）。没有门控（gate 恒为 None）。"""
    arch = "deb"
    ordinal = True

    def __init__(self, dA, dV):
        super().__init__()
        self.enc = AutoModel.from_pretrained(DEB_PATH)
        for p in self.enc.parameters():
            p.requires_grad = False
        self.n_lora = 0
        targets = tuple(CFG["LORA3_TARGETS"])
        for name, mod in list(self.enc.named_modules()):
            for cn, child in list(mod.named_children()):
                full = f"{name}.{cn}" if name else cn
                if isinstance(child, nn.Linear) and ".layer." in "." + full and full.endswith(targets):
                    setattr(mod, cn, GatedLoRALinear(child, CFG["LORA3_R"], CFG["LORA3_ALPHA"], 0, 1.0,
                                                     CFG["LORA3_DROPOUT"], None))
                    self.n_lora += 1
        if self.n_lora == 0:
            raise RuntimeError("DeBERTa 里没有找到可加 LoRA 的线性层，请检查 LORA3_TARGETS")
        Hd = self.enc.config.hidden_size
        self.fc1 = nn.Linear(2 * (Hd + dA + dV), CFG["HIDDEN"])
        self.fc2 = nn.Linear(CFG["HIDDEN"], CFG["FUSE_DIM"])
        self.head_c = nn.Linear(CFG["FUSE_DIM"], 3)
        self.head_o = nn.Linear(CFG["FUSE_DIM"], CFG["NGRID"])
        self.drop = nn.Dropout(CFG["DROPOUT"])
        self.register_buffer("grid", torch.tensor(GRID, dtype=torch.float32), persistent=False)
        self.gate_T = self.gate_F = None
        self.gate_mode = "off"
        self.last_ord = None

    def make_batch(self, ds, idx, ms=None, noise=None):
        return to_batch_deb(ds, idx, ms, noise)

    def encode_text(self, ids, attn, aT):
        with autocast():
            hs = self.enc(input_ids=ids, attention_mask=attn).last_hidden_state
        return masked_mean_std(hs, aT)

    def encode_rows(self, b, ri=None):
        if ri is None:
            return self.encode_text(b["ids"], b["attn"], b["aT"])
        return self.encode_text(b["ids"][ri], b["attn"][ri], b["aT"][ri])

    def gate(self, net, phi, ind):
        return None

    def fuse_full(self, zT, audio, vision, aA, aV):
        z = torch.cat([zT, masked_mean_std(audio, aA), masked_mean_std(vision, aV)], dim=-1)
        with autocast():
            h = self.drop(F.relu(self.fc1(z)))
            h = self.drop(F.relu(self.fc2(h)))
            logits = self.head_c(h)
            ol = self.head_o(h)
        ol = ol.float()
        yexp = (torch.softmax(ol, -1) * self.grid).sum(-1)
        return logits.float(), yexp, h.float(), ol

    def fuse(self, zT, audio, vision, aA, aV, phi=None, indF=None):
        logits, yexp, h, ol = self.fuse_full(zT, audio, vision, aA, aV)
        self.last_ord = ol
        return logits, yexp, h, None

    def forward_batch(self, b):
        return self.fuse_full(self.encode_rows(b), b["audio"], b["vision"], b["aA"], b["aV"])


def ordinal_targets(y, grid_t):
    d = torch.exp(-0.5 * ((y[:, None] - grid_t[None, :]) / CFG["ORD_SIGMA"]) ** 2)
    return d / d.sum(1, keepdim=True)


def soft_macro_f1_loss(logits, c):
    """1 − 宏 F1 的可微近似：TP/FP/FN 用预测概率代替 0/1 计数（批内）。"""
    p = torch.softmax(logits.float(), -1)
    oh = F.one_hot(c, 3).float()
    tp = (p * oh).sum(0); fp = (p * (1 - oh)).sum(0); fn = ((1 - p) * oh).sum(0)
    return 1.0 - (2 * tp / (2 * tp + fp + fn + 1e-6)).mean()


def sym_kl(a, b):
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    return 0.5 * ((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1)).mean()


def deb_param_counts(model):
    cnt = lambda f: int(sum(p.numel() for n, p in model.named_parameters() if f(n)))
    return dict(total=cnt(lambda n: True), backbone_frozen=cnt(lambda n: n.startswith("enc.") and "task_" not in n),
                lora=cnt(lambda n: "task_" in n), heads=cnt(lambda n: n.startswith(("fc1.", "fc2.", "head_c.", "head_o."))),
                n_lora_modules=int(model.n_lora))


def quick_eval_ds(model, ds):
    pr = TorchPredictor(model=model)
    full = pr.predict(ds, np.arange(ds.n))
    m = metrics(ds.c, ds.y, full["probs"], full["yhat"])
    return dict(f1=m["f1"], acc=m["acc"], mae=m["mae"], f1_neu=m["f1_neu"], rec_neu=m["rec_neu"])


def train_deb(data, seed, sel_ds):
    """只训练 LoRA 与头部。选 epoch 的规则（事先定好）：选择区 VS 完整输入上 F1 + Acc − MAE 最大；校准区 VC 不参与。"""
    set_seed(seed)
    tr = data["train"]
    model = DebModel(tr.audio.shape[2], tr.vision.shape[2]).to(DEVICE)
    lora = [p for n, p in model.named_parameters() if "task_" in n]
    head = [p for n, p in model.named_parameters() if n.startswith(("fc1.", "fc2.", "head_c.", "head_o."))]
    for p in lora + head:
        p.requires_grad = True
    params = lora + head
    opt = torch.optim.AdamW([{"params": lora, "lr": CFG["DEB_LR_LORA"]}, {"params": head, "lr": CFG["DEB_LR_HEAD"]}],
                            weight_decay=CFG["DEB_WD"])
    bs = CFG["DEB_BS"]; steps = CFG["DEB_EPOCHS"] * math.ceil(tr.n / bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(CFG["DEB_WARMUP"] * steps)))
    scaler = make_scaler(); cw = class_weights(tr); rng = np.random.default_rng(seed)
    grid_t = _t(GRID)
    best_key, best_state, hist, bad, consec = None, None, [], 0, 0
    log(f"  [deb s{seed}] LoRA 模块 {model.n_lora} 个；可训练参数 {sum(p.numel() for p in params) / 1e6:.2f}M")
    for ep in range(CFG["DEB_EPOCHS"]):
        model.train(); perm = rng.permutation(tr.n); parts = defaultdict(list)
        for st in range(0, tr.n, bs):
            idx = perm[st:st + bs]
            ms = sample_family_masks(tr, idx, rng, p_masked=CFG["AUG_P"])[0] if CFG["AUG_P"] > 0 else None
            b = model.make_batch(tr, idx, ms)
            yc = _t(tr.c[idx], torch.long); yv = _t(tr.y[idx]); tgt = ordinal_targets(yv, grid_t)
            outs = [model.forward_batch(b)]
            if CFG["RDROP_ALPHA"] > 0:
                outs.append(model.forward_batch(b))
            task, comp = [], defaultdict(list)
            for lg, ye, _, ol in outs:
                ce = F.cross_entropy(lg, yc, weight=cw)
                ordl = -(tgt * F.log_softmax(ol, -1)).sum(-1).mean()
                l1 = F.l1_loss(ye, yv)
                sf1 = soft_macro_f1_loss(lg, yc)
                task.append(ce + CFG["W_ORD"] * ordl + CFG["W_L1"] * l1 + CFG["W_SF1"] * sf1)
                for k_, v_ in (("ce", ce), ("ord", ordl), ("l1", l1), ("sf1", sf1)):
                    comp[k_].append(v_.detach())
            loss = sum(task) / len(task)
            if len(outs) == 2:
                rd = 0.5 * (sym_kl(outs[0][0], outs[1][0]) + sym_kl(outs[0][3], outs[1][3]))
                loss = loss + CFG["RDROP_ALPHA"] * rd; comp["rdrop"].append(rd.detach())
            if not torch.isfinite(loss):
                bad += 1; consec += 1; opt.zero_grad(set_to_none=True)
                if consec > 25:
                    raise RuntimeError("DeBERTa 训练连续 25 步出现非有限损失，已停止（可把 DEB_LR_LORA 调小或关闭 AMP）")
                continue
            consec = 0
            _step(opt, sch, scaler, loss, params)
            for k_, v_ in comp.items():
                parts[k_].extend(v_)
            parts["loss"].append(loss.detach())
        ev = quick_eval_ds(model, sel_ds)
        ev.update(epoch=ep + 1, **{f"train_{k}": float(torch.stack(v).float().mean().item()) for k, v in parts.items() if v},
                  nonfinite_skips=bad)
        key = ev["f1"] + ev["acc"] - ev["mae"]
        key = key if np.isfinite(key) else -np.inf
        ev["sel_score"] = key
        hist.append(ev)
        log(f"  [deb s{seed}] ep{ep + 1}: VS F1={ev['f1']:.4f} Acc={ev['acc']:.4f} MAE={ev['mae']:.4f} "
            f"中性F1={ev['f1_neu']:.3f} | 损失 {ev.get('train_loss', float('nan')):.4f}")
        if best_key is None or key > best_key:
            best_key, best_state = key, trainable_state(model)
    load_trainable(model, best_state)
    sel = max(hist, key=lambda h: h["sel_score"])["epoch"]
    for h in hist:
        h["selected"] = int(h["epoch"] == sel)
    log(f"  [deb s{seed}] 选中 epoch {sel}（VS 上 F1+Acc−MAE 最大）")
    return model, hist, best_state


def run_deb(data, sel_ds):
    d = ckpt_dir(data, keys=DEB_KEYS, prefix="deb_", name_re=CKPT_NAME_DEB, cfg_name="train_config_deb.json")
    preds, hists, timing = {}, {}, {}
    have = [s for s in CFG["DEB_SEEDS"] if (d / f"deb_s{s}.pt").exists()]
    log(f"DeBERTa 检查点：{len(have)}/{len(CFG['DEB_SEEDS'])} 个已存在")
    for s in CFG["DEB_SEEDS"]:
        ck = d / f"deb_s{s}.pt"
        if ck.exists():
            log(f"读取已有检查点 {ck.name}"); state = _load_state(ck)
        else:
            log(f"==== 训练 deb seed {s} ====")
            t1 = time.time()
            try:
                model, hist, state = train_deb(data, s, sel_ds)
            except Exception as e:                     # 某个种子失败（如显存不足）只跳过这个种子，已完成的种子照常使用
                log(f"!!! deb s{s} 训练失败：{e!r}（跳过这个种子）")
                S3.setdefault("errors", []).append(f"deb s{s}: {e!r}")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            if "deb_params" not in S3:
                S3["deb_params"] = deb_param_counts(model); log("DeBERTa 参数量：", S3["deb_params"])
            torch.save(state, ck)
            hists[("deb", s)] = hist; timing[f"deb_s{s}"] = time.time() - t1
            pd.DataFrame(hist).to_csv(OUT / "runs" / f"history_deb_s{s}.csv", index=False)
            log(f"  用时 {time.time() - t1:.0f}s")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        preds[("deb", s)] = TorchPredictor(state, "off", key=("deb", s), arch="deb")
    save_json(timing, OUT / "tables" / "train_time_deb_seconds.json")
    S3["deb_ckpt_dir"] = str(d)
    return preds, hists

# %% [evaluate] 6. 统一评估：完整输入 + PM 面板 + LH 压力面板 + SC 散点面板
PANEL_KINDS = ("PM", "LH", "SC", "MS")


def evaluate_predictor(pred, ds, tag, save=True):
    full = pred.predict(ds, np.arange(ds.n))
    res = dict(full=full)
    if ds.has_label:
        res["complete"] = metrics(ds.c, ds.y, full["probs"], full["yhat"])
    for kind in PANEL_KINDS:
        preds = [(nm, cb, r, pos, pred.predict(ds, np.arange(ds.n), ms)) for nm, cb, r, pos, ms in PANELS[ds.name][kind]]
        res[kind] = preds
        if ds.has_label:
            df = panel_table(ds, preds)
            res[kind + "_df"] = df
            res[kind + "_sum"] = panel_summary(df, res["complete"])
            if save:
                df.to_csv(OUT / "runs" / f"panel_{kind}_{tag}.csv", index=False)
    return res


def evaluate_all(preds, ds):
    results, rows = {}, []
    for (cfg_name, seed), pr in preds.items():
        tag = f"{ds.name}_{cfg_name}_s{seed}"
        r = evaluate_predictor(pr, ds, tag)
        results[(cfg_name, seed)] = r
        row = dict(split=ds.name, config=cfg_name, seed=seed)
        row.update({f"complete_{k}": v for k, v in r["complete"].items()})
        for kind in PANEL_KINDS:
            row.update({f"{kind}_{k}": v for k, v in r[kind + "_sum"].items()})
        rows.append(row)
        log(f"  {tag}: 完整F1={r['complete']['f1']:.4f} MAE={r['complete']['mae']:.4f} | "
            f"PM均F1={r['PM_sum']['avg_f1']:.4f} 最差={r['PM_sum']['worst_f1']:.4f} | "
            f"LH均F1={r['LH_sum']['avg_f1']:.4f} | SC均F1={r['SC_sum']['avg_f1']:.4f} | MS均F1={r['MS_sum']['avg_f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"summary_runs_{ds.name}.csv", index=False)
    num = [c for c in df.columns if c not in ("split", "config", "seed") and not c.endswith("worst_bucket")]
    agg = df.groupby("config")[num].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(OUT / "tables" / f"summary_by_config_{ds.name}.csv")
    return results, df, agg


SEED_METRICS = ["complete_f1", "complete_acc", "complete_mae", "complete_corr", "PM_avg_f1", "PM_avg_mae", "PM_worst_f1",
                "LH_avg_f1", "LH_avg_mae", "SC_avg_f1", "SC_avg_mae", "MS_avg_f1", "MS_avg_mae"]


PAIRS = [("deb", "base"), ("deb", "joint"), ("joint", "base"), ("mg", "base"), ("mg", "joint"), ("mg2", "base"),
         ("mg2", "mg"), ("mg2", "joint"), ("mg", "mg_nogate"), ("mg", "mg_nokd"), ("mg", "mg_scatter")]


def seed_level_diffs(df, split):
    """同种子配对：每个种子上 A − B 的差值，给出均值、标准差与 t 分布 95% 区间（种子层面的不确定性）。
    注意：这是“5 个种子的配对差”的 t 区间，只反映训练随机性；与按视频重抽样的区间（反映样本不确定性）不是一回事。"""
    from scipy.stats import t as tdist
    pairs = PAIRS
    rows = []
    for a, b in pairs:
        A = df[df.config == a].set_index("seed"); B = df[df.config == b].set_index("seed")
        seeds = sorted(set(A.index) & set(B.index))
        if not seeds:
            continue
        for met in SEED_METRICS:
            d = (A.loc[seeds, met] - B.loc[seeds, met]).values.astype(float)
            k = len(d); mu = float(d.mean()); sd = float(d.std(ddof=1)) if k > 1 else float("nan")
            half = float(tdist.ppf(0.975, k - 1) * sd / math.sqrt(k)) if k > 1 else float("nan")
            rows.append(dict(split=split, compare=f"{a} - {b}", metric=met, n_seeds=k, mean=mu, sd=sd,
                             lo=mu - half, hi=mu + half, n_positive=int((d > 0).sum()), per_seed=[round(float(x), 5) for x in d]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tables" / f"seed_diffs_{split}.csv", index=False)
    return out


def bootstrap_comparisons(results, ds, seed, pairs=None, tag=""):
    """视频配对重抽样（样本层面的不确定性）：同一批样本上 A − B，按源视频有放回抽样 BOOT_REPS 次。
    pairs 里的元素可以是 (A, B)（都取 seed 这个种子）或 ((A, sA), (B, sB))（比如集成的“种子”记为 0）。"""
    rows = []
    for pr in (pairs or PAIRS):
        if isinstance(pr[0], tuple):
            (a, sa), (b, sb) = pr
        else:
            (a, sa), (b, sb) = (pr[0], seed), (pr[1], seed)
        ka, kb = (a, sa), (b, sb)
        if ka not in results or kb not in results:
            continue
        for kind in ("complete",) + PANEL_KINDS:
            if kind == "complete":
                pa = [results[ka]["full"]]; pb = [results[kb]["full"]]
            else:
                pa = [x[4] for x in results[ka][kind]]; pb = [x[4] for x in results[kb][kind]]
            bs = paired_bootstrap(ds, pa, pb, seed=int(hashlib.md5(f"{a}|{b}|{kind}".encode()).hexdigest()[:6], 16))
            nm = lambda x, s_: x if s_ == 0 else f"{x}_s{s_}"
            for met in ("avg_f1", "avg_acc", "avg_mae", "worst_f1"):
                if kind == "complete" and met == "worst_f1":
                    continue
                rows.append(dict(compare=f"{nm(a, sa)} - {nm(b, sb)}", panel=kind, metric=met, **bs[met], groups=bs["groups"]))
    df = pd.DataFrame(rows)
    if len(df):
        df["excludes_zero"] = (df.lo > 0) | (df.hi < 0)
    df.to_csv(OUT / "tables" / f"bootstrap_{ds.name}{tag}.csv", index=False)
    log(f"视频配对重抽样（{ds.name}{tag}）：\n" + (df.to_string() if len(df) else "无"))
    return df


def law_tables(results, key, ds, name):
    """缺失规律：按被遮蔽模态、比例、位置分组平均；另给 “模态 × 比例” 与 “模态 × 位置” 交叉表。"""
    tabs = []
    for kind in PANEL_KINDS:
        df = results[key][kind + "_df"]
        facs = ["combo", "ratio"] + (["pos"] if kind in ("PM", "LH") else [])
        for fac in facs:
            g = df.groupby(fac)[["f1", "acc", "mae", "corr", "gateF", "gateT"]].mean().reset_index()
            g.insert(0, "factor", fac); g.insert(0, "panel", kind); g = g.rename(columns={fac: "level"})
            tabs.append(g)
        for f2 in (["ratio", "pos"] if kind in ("PM", "LH") else ["ratio"]):
            g = df.groupby(["combo", f2])[["f1", "mae"]].mean().reset_index()
            g["level"] = g["combo"] + "|" + g[f2].astype(str); g["factor"] = f"combo×{f2}"; g["panel"] = kind
            tabs.append(g[["panel", "factor", "level", "f1", "mae"]])
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(OUT / "tables" / f"law_{name}_{ds.name}.csv", index=False)
    return tab


def extended_scan(pred, ds, name):
    """扩展扫描：比例扩展到 1.0（整模态删空），看 “全有或全无” 现象。"""
    panel = build_panel(ds, "PM", ratios=CFG["EXT_RATIOS"], allow_clear=True)
    preds = [(nm, cb, r, pos, pred.predict(ds, np.arange(ds.n), ms)) for nm, cb, r, pos, ms in panel]
    df = panel_table(ds, preds)
    df.to_csv(OUT / "tables" / f"ext_scan_{name}_{ds.name}.csv", index=False)
    return df


def gate_analysis(results, key, ds, tag=""):
    """门控值与实际缺失程度的关系：门控能否读懂 “缺得有多严重”。"""
    rows = []
    for kind in ("PM", "SC", "MS"):
        for (nm, cb, r, pos, ms), (_, _, _, _, pr) in zip(PANELS[ds.name][kind], results[key][kind]):
            aT, aA, aV, _ = apply_masks(ds, np.arange(ds.n), ms)
            phi, _, _ = descriptors(aT, aA, aV, ref_rows(ds, np.arange(ds.n)))
            rows.append(pd.DataFrame(dict(panel=kind, bucket=nm, combo=cb, ratio=r, pos=pos,
                                          miss_T=phi[:, 0], miss_A=phi[:, 3], miss_V=phi[:, 6],
                                          gap_T=phi[:, 1], gap_A=phi[:, 4], gap_V=phi[:, 7],
                                          gateT=pr["gT"], gateF=pr["gF"])))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "tables" / f"gate_analysis_{tag}{ds.name}.csv", index=False)
    mm = df[["miss_T", "miss_A", "miss_V"]].max(1)
    corr = dict(gateF_vs_missmax=float(np.corrcoef(mm, df.gateF)[0, 1]) if df.gateF.std() > 1e-9 else None,
                gateT_vs_missT=float(np.corrcoef(df.miss_T, df.gateT)[0, 1]) if df.gateT.std() > 1e-9 else None,
                gateF_mean_by_combo=df.groupby("combo").gateF.mean().round(4).to_dict())
    return df, corr


# %% [rules] 7. 规则层：强度校准、中性判定、极性—强度一致性（验证集按视频分组交叉验证后再决定是否启用）
def fit_calib(yhat, y):
    a, b = np.polyfit(yhat.astype(np.float64), y.astype(np.float64), 1)
    return float(a), float(b)


def apply_calib(yhat, ab):
    return np.clip(ab[0] * yhat + ab[1], -3, 3) if ab is not None else yhat


def neutral_rule(probs, ycal, prm):
    """中性判定：非中性但概率不高且强度接近 0 → 改判中性；判为中性但强度明显偏离 0 → 按符号改判。"""
    c = probs.argmax(1)
    if prm is None:
        return c
    th, tau, tau2 = prm
    pmax = probs.max(1)
    c2 = c.copy()
    c2[(c != 1) & (pmax < th) & (np.abs(ycal) < tau)] = 1
    if tau2 is not None:
        m = (c == 1) & (np.abs(ycal) > tau2)
        c2[m] = np.where(ycal[m] > 0, 2, 0)
    return c2


def fit_neutral(probs, ycal, c_true):
    best = (f1_from_cm(conf_mat(c_true, probs.argmax(1)))[0], None)
    for th in np.arange(0.35, 0.91, 0.05):
        for tau in np.arange(0.05, 1.01, 0.05):
            for tau2 in [None] + list(np.arange(0.3, 1.51, 0.1)):
                f = f1_from_cm(conf_mat(c_true, neutral_rule(probs, ycal, (th, tau, tau2))))[0]
                if f > best[0] + 1e-9:
                    best = (f, (float(th), float(tau), None if tau2 is None else float(tau2)))
    return best[1]


def coherent_intensity(c, y, eps):
    """题面定义：强度 [-3,0) 负向、0 中性、(0,3] 正向。让输出的强度与极性自洽。"""
    y = y.copy()
    y[c == 1] = 0.0
    y[c == 0] = np.minimum(y[c == 0], -eps)
    y[c == 2] = np.maximum(y[c == 2], eps)
    return y


def _variants(probs, yhat, calib, neutral, eps):
    ycal = apply_calib(yhat, calib)
    c0 = probs.argmax(1); cn = neutral_rule(probs, ycal, neutral)
    return {"raw": (c0, yhat), "calib": (c0, ycal), "calib+neutral": (cn, ycal),
            "raw+coherent": (c0, coherent_intensity(c0, yhat, eps)),
            "calib+coherent": (c0, coherent_intensity(c0, ycal, eps)),
            "calib+neutral+coherent": (cn, coherent_intensity(cn, ycal, eps))}


def rules_cv(full, pm_preds, ds, eps):
    """按源视频分组 K 折：折内拟合（只用完整输入的预测），折外评估完整输入与 PM 面板上的效果。"""
    vids = np.array(ds.videos); uv = np.unique(vids)
    rng = np.random.default_rng(2026); rng.shuffle(uv)
    folds = np.array_split(uv, CFG["RULE_CV_FOLDS"])
    names = list(_variants(full["probs"][:1], full["yhat"][:1], None, None, eps).keys())
    oof_c = {v: np.zeros(ds.n, np.int64) for v in names}; oof_y = {v: np.zeros(ds.n, np.float32) for v in names}
    pm_c = [{v: np.zeros(ds.n, np.int64) for v in names} for _ in pm_preds]
    pm_y = [{v: np.zeros(ds.n, np.float32) for v in names} for _ in pm_preds]
    for fv in folds:
        te = np.isin(vids, fv); trm = ~te
        ab = fit_calib(full["yhat"][trm], ds.y[trm])
        prm = fit_neutral(full["probs"][trm], apply_calib(full["yhat"][trm], ab), ds.c[trm])
        for v, (c, y) in _variants(full["probs"][te], full["yhat"][te], ab, prm, eps).items():
            oof_c[v][te] = c; oof_y[v][te] = y
        for b, pr in enumerate(pm_preds):
            for v, (c, y) in _variants(pr["probs"][te], pr["yhat"][te], ab, prm, eps).items():
                pm_c[b][v][te] = c; pm_y[b][v][te] = y
    rows = []
    for v in names:
        mc = metrics(ds.c, ds.y, full["probs"], oof_y[v], c_pred=oof_c[v])
        pmm = [metrics(ds.c, ds.y, pr["probs"], pm_y[b][v], c_pred=pm_c[b][v]) for b, pr in enumerate(pm_preds)]
        rows.append(dict(variant=v, complete_f1=mc["f1"], complete_acc=mc["acc"], complete_mae=mc["mae"],
                         complete_corr=mc["corr"], complete_rec_neu=mc["rec_neu"],
                         PM_avg_f1=float(np.mean([m["f1"] for m in pmm])) if pmm else np.nan,
                         PM_avg_mae=float(np.mean([m["mae"] for m in pmm])) if pmm else np.nan))
    return pd.DataFrame(rows).set_index("variant")


class RuleLayer:
    def __init__(self, eps):
        self.calib = None; self.neutral = None; self.coherent = False; self.eps = eps; self.info = {}

    def fit(self, full, pm_preds, ds):
        cv = rules_cv(full, pm_preds, ds, self.eps)
        cv.to_csv(OUT / "tables" / "rules_cv.csv")
        log("规则层交叉验证（折外）：\n" + cv.round(4).to_string())
        raw, cal, neu = cv.loc["raw"], cv.loc["calib"], cv.loc["calib+neutral"]
        use_cal = (cal.complete_mae < raw.complete_mae - 1e-4) and not (cal.PM_avg_mae > raw.PM_avg_mae + 0.002)
        base = cal if use_cal else raw
        use_neu = (neu.complete_f1 >= base.complete_f1 + CFG["RULE_MIN_GAIN"]) and not (neu.PM_avg_f1 < base.PM_avg_f1 - 0.002)
        key = ("calib" if use_cal else "raw") if not use_neu else "calib+neutral"
        coh_key = {"raw": "raw+coherent", "calib": "calib+coherent", "calib+neutral": "calib+neutral+coherent"}[key]
        use_coh = cv.loc[coh_key].complete_mae <= cv.loc[key].complete_mae + CFG["COHERENT_MAX_MAE_LOSS"]
        if use_cal or use_neu:
            self.calib = fit_calib(full["yhat"], ds.y)
        if use_neu:
            self.neutral = fit_neutral(full["probs"], apply_calib(full["yhat"], self.calib), ds.c)
        self.use_cal_output = bool(use_cal or use_neu)   # 与交叉验证中评估的变体保持一致
        self.coherent = bool(use_coh)
        chosen = coh_key if use_coh else key
        self.info = dict(use_calib=bool(use_cal), use_neutral=bool(use_neu), use_coherent=bool(use_coh),
                         chosen_variant=chosen, calib=self.calib, neutral=self.neutral, eps=self.eps,
                         cv=cv.round(5).reset_index().to_dict("records"))
        log("规则层决定：", {k: v for k, v in self.info.items() if k != "cv"})
        return self

    def apply(self, probs, yhat):
        ycal = apply_calib(yhat, self.calib)
        c = neutral_rule(probs, ycal, self.neutral)
        y = ycal if self.use_cal_output else yhat
        y_out = coherent_intensity(c, y, self.eps) if self.coherent else y
        return c, y_out, y


def review_flags(full, shap_df):
    """人工复核标记：最大类别概率 < 阈值，或分类主导模态不是文本。"""
    pmax = full["probs"].max(1)
    return (pmax < CFG["REVIEW_PMAX"]) | (shap_df.dom_cls.values != "T")


def review_rule_eval(full, shap_df, ds):
    flag = review_flags(full, shap_df)
    err = full["probs"].argmax(1) != ds.c
    nt = shap_df.dom_cls.values != "T"
    lowp = full["probs"].max(1) < CFG["REVIEW_PMAX"]
    r = dict(coverage=float(flag.mean()),
             err_rate_all=float(err.mean()),
             err_rate_flagged=float(err[flag].mean()) if flag.any() else None,
             err_rate_unflagged=float(err[~flag].mean()) if (~flag).any() else None,
             share_of_errors_caught=float(err[flag].sum() / max(1, err.sum())),
             n_lowp=int(lowp.sum()), err_rate_lowp=float(err[lowp].mean()) if lowp.any() else None,
             n_nontext_dominant=int(nt.sum()), err_rate_nontext=float(err[nt].mean()) if nt.any() else None,
             err_rate_text=float(err[~nt].mean()) if (~nt).any() else None)
    return flag, r


# %% [q3] 8. 问题三：精确 Shapley、逐位置重要性、删除检验、稳定性、解释卡
SUBSETS = [(), ("T",), ("A",), ("V",), ("T", "A"), ("T", "V"), ("A", "V"), ("T", "A", "V")]


def _remove_except(ds, idx, keep):
    n = len(idx)
    return {m: (np.zeros((n, ds.T), bool) if m in keep else ds.o(m)[idx].copy()) for m in MODS}


def exact_shapley(pred, ds, idx=None):
    """三个模态只有 8 个子集，Shapley 值可以精确计算（不用抽样近似）；分类头与强度头各算一套。"""
    idx = np.arange(ds.n) if idx is None else np.asarray(idx)
    n = len(idx)
    full = pred.predict(ds, idx)
    cstar = full["probs"].argmax(1)
    vc, vr = {}, {}
    for S in SUBSETS:
        p = full if len(S) == 3 else pred.predict(ds, idx, _remove_except(ds, idx, S))
        vc[S] = p["probs"][np.arange(n), cstar]; vr[S] = p["yhat"]
    phi = {h: {m: np.zeros(n) for m in MODS} for h in ("cls", "reg")}
    for m in MODS:
        others = [x for x in MODS if x != m]
        for k in range(0, 3):
            for S in combinations(others, k):
                w = math.factorial(k) * math.factorial(2 - k) / 6.0
                Sm = tuple(x for x in MODS if x in S or x == m)
                S0 = tuple(x for x in MODS if x in S)
                phi["cls"][m] += w * (vc[Sm] - vc[S0])
                phi["reg"][m] += w * (vr[Sm] - vr[S0])
    rows = dict(id=[ds.ids[i] for i in idx], cstar=cstar, p_cstar=vc[("T", "A", "V")], yhat=vr[("T", "A", "V")],
                v0_cls=vc[()], v0_reg=vr[()])
    for h in ("cls", "reg"):
        mat = np.stack([phi[h][m] for m in MODS], 1)
        for j, m in enumerate(MODS):
            rows[f"phi_{h}_{m}"] = mat[:, j]
        absm = np.abs(mat); tot = absm.sum(1, keepdims=True)
        share = np.where(tot > 1e-9, absm / np.maximum(tot, 1e-12), np.nan)
        for j, m in enumerate(MODS):
            rows[f"share_{h}_{m}"] = share[:, j]
        srt = np.sort(np.nan_to_num(share), 1)
        rows[f"margin_{h}"] = srt[:, -1] - srt[:, -2]
        rows[f"dom_{h}"] = np.array(MODS)[absm.argmax(1)]
        v_full = vc[("T", "A", "V")] if h == "cls" else vr[("T", "A", "V")]
        v_0 = vc[()] if h == "cls" else vr[()]
        rows[f"eff_err_{h}"] = np.abs(mat.sum(1) - (v_full - v_0))
        # Shapley 交互指数（n=3 时可精确算）：I_ij = ½[v(ij)−v(i)−v(j)+v(∅)] + ½[v(ijk)−v(ik)−v(jk)+v(k)]
        v = vc if h == "cls" else vr
        key = lambda *ms: tuple(x for x in MODS if x in ms)
        for i, j in (("T", "A"), ("T", "V"), ("A", "V")):
            k = [x for x in MODS if x not in (i, j)][0]
            rows[f"I_{h}_{i}{j}"] = 0.5 * (v[key(i, j)] - v[key(i)] - v[key(j)] + v[()]) + \
                0.5 * (v[key(i, j, k)] - v[key(i, k)] - v[key(j, k)] + v[key(k)])
        rows[f"I_{h}_TAV"] = (v[key("T", "A", "V")] - v[key("T", "A")] - v[key("T", "V")] - v[key("A", "V")]
                              + v[key("T")] + v[key("A")] + v[key("V")] - v[()])
    for S in SUBSETS:
        nm = "".join(S) or "none"
        rows[f"v_cls_{nm}"] = vc[S]; rows[f"v_reg_{nm}"] = vr[S]
    return pd.DataFrame(rows), full


def local_importance(pred, ds, i, cstar, noise=None, mods=MODS, freeze=False):
    """样本 i 的每个模态：逐位置遮蔽以该位置为中心、宽 LOCAL_WIN 的窗口，得分 = |Δ强度| + |Δp(c*)|。"""
    h = CFG["LOCAL_WIN"] // 2
    base = pred.predict(ds, [i], None, noise)
    b_p = base["probs"][0, cstar]; b_y = base["yhat"][0]
    res = {}
    for m in mods:
        P = np.flatnonzero(ds.o(m)[i])
        if len(P) == 0:
            res[m] = (P, np.zeros(0)); continue
        R = len(P)
        ms = {x: np.zeros((R, ds.T), bool) for x in MODS}
        for r_, t in enumerate(P):
            ms[m][r_, P[(P >= t - h) & (P <= t + h)]] = True
        p = pred.predict(ds, np.full(R, i), ms, noise, freeze_gate=freeze)
        res[m] = (P, np.abs(p["yhat"] - b_y) + np.abs(p["probs"][:, cstar] - b_p))
    return res


def faithfulness(pred, ds, idx, imps, cstars, rng, freeze=False):
    """删除检验与保留检验：高分位置 vs 低分位置 vs 随机位置（同一样本内重复 FAITH_RAND 次取平均）。"""
    rows = []
    for i, cs in zip(idx, cstars):
        base = pred.predict(ds, [i])["probs"][0, cs]
        for m in MODS:
            P, imp = imps[i][m]
            if len(P) < 2:
                continue
            order = np.argsort(-imp)
            for bud in CFG["FAITH_BUDGETS"]:
                k = max(1, int(round(bud * len(P))))
                sets = [("top", P[order[:k]]), ("low", P[order[-k:]])]
                sets += [("rand", rng.choice(P, k, replace=False)) for _ in range(CFG["FAITH_RAND"])]
                R = 2 * len(sets)
                ms = {x: np.zeros((R, ds.T), bool) for x in MODS}
                for j, (_, S) in enumerate(sets):
                    ms[m][2 * j, S] = True
                    ms[m][2 * j + 1, np.setdiff1d(P, S)] = True
                p = pred.predict(ds, np.full(R, i), ms, freeze_gate=freeze)["probs"][:, cs]
                d_del = base - p[0::2]; d_keep = base - p[1::2]
                rows.append(dict(id=ds.ids[i], modality=m, budget=bud, k=k, n_pos=len(P),
                                 del_top=d_del[0], del_low=d_del[1], del_rand=float(d_del[2:].mean()),
                                 keep_top=d_keep[0], keep_low=d_keep[1], keep_rand=float(d_keep[2:].mean())))
    return pd.DataFrame(rows)


def words_of(ds, i, tokenizer):
    """子词位置 → 整词（处理 ## 续接）。返回每个位置对应的整词（非词位置为空串）与子词序列。"""
    ids = [int(x) for x in ds.input_ids[i]]
    toks = tokenizer.convert_ids_to_tokens(ids) if tokenizer is not None else [f"<{x}>" for x in ids]
    words = [""] * ds.T
    spans = []
    for t in np.flatnonzero(ds.W[i]):
        tk = toks[t]
        if tk.startswith("##") and spans and spans[-1][-1] == t - 1:
            spans[-1].append(t)
        else:
            spans.append([t])
    # v3：把缩写拆出的三段（如 don ' t、it ' s）并回一个词，证据与对齐都按整词处理
    merged, j = [], 0
    while j < len(spans):
        if (j + 2 < len(spans) and len(spans[j + 1]) == 1 and toks[spans[j + 1][0]] == "'"
                and toks[spans[j + 2][0]] in ("s", "t", "re", "ve", "ll", "d", "m")
                and spans[j + 1][0] == spans[j][-1] + 1 and spans[j + 2][0] == spans[j + 1][0] + 1):
            merged.append(spans[j] + spans[j + 1] + spans[j + 2]); j += 3
        else:
            merged.append(spans[j]); j += 1
    spans = merged
    for sp in spans:
        w = "".join(toks[t][2:] if toks[t].startswith("##") else toks[t] for t in sp)
        for t in sp:
            words[t] = w
    return words, toks, spans


_ALNUM = re.compile(r"[0-9A-Za-z]")


def evidence_of(imp_res, words, mod, topk, spans=None):
    """v3：按整词汇总——每个词取其各位置得分的最大值；纯标点词不作为证据；取得分最高的 topk 个词，相邻词合并成片段。
    spans：words_of 返回的整词位置列表；为 None 时退回按位置处理（兼容旧调用）。"""
    P, imp = imp_res[mod]
    if len(P) == 0:
        return []
    sc = dict(zip(P.tolist(), imp.tolist()))
    units = []
    if spans:
        inw = set()
        for k, sp in enumerate(spans):
            ps = [t for t in sp if t in sc]
            inw.update(sp)
            if ps:
                units.append(dict(k=k, positions=ps, word=words[sp[0]], score=max(sc[t] for t in ps)))
        for t in P.tolist():                      # 词范围外的位置（极少见）：单独成一个单元
            if t not in inw:
                units.append(dict(k=10_000 + t, positions=[t], word="", score=sc[t]))
    else:
        units = [dict(k=t, positions=[t], word=words[t], score=sc[t]) for t in P.tolist()]
    n_all = len(units)
    units = [u for u in units if (not u["word"]) or _ALNUM.search(u["word"])]
    n_punct = n_all - len(units)
    if not units:
        return []
    top = sorted(sorted(units, key=lambda u: -u["score"])[:topk], key=lambda u: u["k"])
    groups, cur = [], [top[0]]
    for u in top[1:]:
        if u["k"] == cur[-1]["k"] + 1:
            cur.append(u)
        else:
            groups.append(cur); cur = [u]
    groups.append(cur)
    out = []
    for g in groups:
        out.append(dict(modality=mod, positions=[int(t) for u in g for t in u["positions"]],
                        words=" ".join(u["word"] for u in g if u["word"]), score=float(max(u["score"] for u in g)),
                        n_punct_excluded=int(n_punct)))
    return sorted(out, key=lambda e: -e["score"])


def explain_set(pred, ds, idx, tokenizer, rng, shap_df=None, noise_check=True, freeze=False):
    """对一组样本输出解释卡（含逐位置重要性），并做删除检验和稳定性检验。"""
    from scipy.stats import spearmanr
    if shap_df is None:
        shap_df, _ = exact_shapley(pred, ds, idx)
    sd = shap_df.set_index("id")
    cards, imps, spear = [], {}, []
    for i in idx:
        r = sd.loc[ds.ids[i]]
        cs = int(r.cstar)
        imp = local_importance(pred, ds, i, cs, freeze=freeze)
        imps[i] = imp
        if noise_check:
            imp_n = local_importance(pred, ds, i, cs, noise=(np.random.default_rng(int(i)), CFG["STAB_NOISE"]), freeze=freeze)
            for m in MODS:
                a, b = imp[m][1], imp_n[m][1]
                if len(a) >= 3 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    spear.append(dict(id=ds.ids[i], modality=m, rho=float(spearmanr(a, b).correlation)))
        words, toks, wspans = words_of(ds, i, tokenizer)
        card = dict(id=ds.ids[i], pred_class=CLASS_NAMES[cs], p_pred=float(r.p_cstar), intensity_raw=float(r.yhat),
                    shapley_cls={m: float(r[f"phi_cls_{m}"]) for m in MODS},
                    shapley_reg={m: float(r[f"phi_reg_{m}"]) for m in MODS},
                    share_cls={m: float(r[f"share_cls_{m}"]) for m in MODS},
                    share_reg={m: float(r[f"share_reg_{m}"]) for m in MODS},
                    dominant_cls=r.dom_cls, dominant_reg=r.dom_reg, margin_cls=float(r.margin_cls),
                    evidence={m: evidence_of(imp, words, m, CFG["EVID_TOPK"], wspans) for m in MODS},
                    local_importance={m: dict(positions=[int(t) for t in imp[m][0]],
                                              words=[words[t] for t in imp[m][0]],
                                              score=[float(s) for s in imp[m][1]]) for m in MODS})
        cards.append(card)
    faith = faithfulness(pred, ds, idx, imps, [int(sd.loc[ds.ids[i]].cstar) for i in idx], rng, freeze=freeze)
    stab = pd.DataFrame(spear)
    return cards, faith, stab, imps


def pick_explain_valid(ds, n_total):
    """验证集解释样本：按真实类别分层，用 SHA-256(盐+样本号) 排序选取，不看模型预测。"""
    per = n_total // 3
    out = []
    for c in range(3):
        cand = [i for i in range(ds.n) if ds.c[i] == c]
        cand.sort(key=lambda i: hashlib.sha256(("gmcm2026E-" + ds.ids[i]).encode()).hexdigest())
        out += cand[:per]
    return np.array(sorted(out))

# %% [analysis] 9. 补充检验：完整输入等价性、训练缺失族覆盖性、错误归因、效率
def native_indicators(ds, idx=None):
    idx = np.arange(ds.n) if idx is None else np.asarray(idx)
    aT, aA, aV, _ = apply_masks(ds, idx, None)
    phi, indT, indF = descriptors(aT, aA, aV, ref_rows(ds, idx))
    return phi, indT, indF


def identity_check(results, ds, cfgs=("mg", "mg2")):
    """结构保证的实测：无原生缺失的样本上，MG-LoRA（v1/v2）与同种子阶段一模型的输出应当逐位相同。"""
    _, _, indF = native_indicators(ds)
    clean = indF == 0
    rows = []
    for (cfg_name, seed) in results:
        if cfg_name not in cfgs or ("base", seed) not in results:
            continue
        a, b = results[(cfg_name, seed)]["full"], results[("base", seed)]["full"]
        dp = np.abs(a["probs"] - b["probs"]).max(1); dy = np.abs(a["yhat"] - b["yhat"])
        rows.append(dict(split=ds.name, config=cfg_name, seed=seed, n_clean=int(clean.sum()), n_native_missing=int((~clean).sum()),
                         max_abs_dprob_clean=float(dp[clean].max()) if clean.any() else None,
                         max_abs_dy_clean=float(dy[clean].max()) if clean.any() else None,
                         mean_abs_dprob_native_missing=float(dp[~clean].mean()) if (~clean).any() else None,
                         pred_changed_native_missing=int((a["probs"].argmax(1) != b["probs"].argmax(1))[~clean].sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"identity_check_{ds.name}.csv", index=False)
    return df


def family_coverage(data, n_draw=20):
    """覆盖性检查（事后，只读附件3的缺失描述量，不参与训练与选择）：
    附件3 的缺失比例、最长缺口是否落在训练缺失族在验证集上生成的分布范围内。"""
    if "att3" not in data:
        return None
    va = data["valid"]; rng = np.random.default_rng(7)
    fam = []
    for _ in range(n_draw):
        idx = np.arange(va.n)
        ms, masked = sample_family_masks(va, idx, rng, 1.0)
        aT, aA, aV, _ = apply_masks(va, idx, ms)
        fam.append(descriptors(aT, aA, aV, ref_rows(va, idx))[0][masked])
    fam = np.concatenate(fam)
    phi3, _, _ = native_indicators(data["att3"])
    names = ["frac_T", "gap_T", "any_T", "frac_A", "gap_A", "any_A", "frac_V", "gap_V", "any_V"]
    rows = []
    for j, nm in enumerate(names):
        if nm.startswith("any"):
            rows.append(dict(feature=nm, att3_mean=float(phi3[:, j].mean()), family_mean=float(fam[:, j].mean())))
            continue
        sel = fam[:, j][fam[:, j] > 0]
        lo, hi = (np.percentile(sel, 1), np.percentile(sel, 99)) if len(sel) else (0, 0)
        v = phi3[:, j][phi3[:, j] > 0]
        rows.append(dict(feature=nm, att3_n_nonzero=int(len(v)), att3_min=float(v.min()) if len(v) else None,
                         att3_max=float(v.max()) if len(v) else None, family_p01=float(lo), family_p99=float(hi),
                         att3_share_inside=float(((v >= lo) & (v <= hi)).mean()) if len(v) else None,
                         att3_quantile_in_family=[float((sel <= x).mean()) for x in v] if len(v) else []))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "att3_family_coverage.csv", index=False)
    return df


def _bins(x, edges, labels):
    return pd.cut(x, edges, labels=labels, include_lowest=True).astype(str)


def error_attribution(ds, full, shap_df, tag, c_pred=None, y_pred=None):
    """验证集错误归因：按真实类别、强度大小、句长、原生缺失、主导模态、置信度分组看错误率与 MAE。"""
    c_pred = full["probs"].argmax(1) if c_pred is None else c_pred
    y_pred = full["yhat"] if y_pred is None else y_pred
    pmax = full["probs"].max(1)
    _, _, indF = native_indicators(ds)
    df = pd.DataFrame(dict(id=ds.ids, video=ds.videos, y=ds.y, c=ds.c, c_pred=c_pred, y_pred=y_pred, pmax=pmax,
                           words=ds.W.sum(1), native_missing=np.where(indF > 0, "yes", "no"),
                           raw_text=ds.raw_text))
    df["err"] = (df.c != df.c_pred).astype(int)
    df["abs_err_y"] = np.abs(df.y - df.y_pred)
    df["true_class"] = [CLASS_NAMES[k] for k in df.c]
    df["pred_class"] = [CLASS_NAMES[k] for k in df.c_pred]
    df["abs_y_bin"] = np.where(df.y == 0, "=0", _bins(np.abs(df.y), [0, 0.5, 1, 2, 3.01], ["(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]"]))
    df["len_bin"] = _bins(df.words, [0, 10, 20, 30, 100], ["<=10", "11-20", "21-30", ">30"])
    df["conf_bin"] = _bins(df.pmax, [0, 0.5, 0.7, 0.9, 1.0], ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])
    if shap_df is not None:
        sd = shap_df.set_index("id").loc[df.id]
        df["dom_cls"] = sd.dom_cls.values; df["margin_cls"] = sd.margin_cls.values
        df["share_T"] = sd.share_cls_T.values
        df["dom_group"] = np.where(df.dom_cls == "T", "text", "non-text")
    order = dict(true_class=CLASS_NAMES, abs_y_bin=["=0", "(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]"],
                 len_bin=["<=10", "11-20", "21-30", ">30"], conf_bin=["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"],
                 native_missing=["no", "yes"], dom_group=["text", "non-text"])
    rows = []
    for fac in ["true_class", "abs_y_bin", "len_bin", "native_missing", "conf_bin"] + (["dom_group"] if shap_df is not None else []):
        groups = dict(list(df.groupby(fac)))
        for lv in [x for x in order[fac] if x in groups] + [x for x in groups if x not in order[fac]]:
            g = groups[lv]
            rows.append(dict(factor=fac, level=lv, n=len(g), share=len(g) / len(df), err_rate=g.err.mean(),
                             mae=g.abs_err_y.mean(), mean_pmax=g.pmax.mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tables" / f"error_attribution_{tag}.csv", index=False)
    cm = pd.DataFrame(conf_mat(ds.c, c_pred), index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES])
    cm.to_csv(OUT / "tables" / f"confusion_{tag}.csv")
    wrong = df[df.err == 1].copy()
    wrong["wrong_conf"] = wrong.pmax
    cols = ["id", "true_class", "pred_class", "y", "y_pred", "pmax", "words"] + (["dom_cls", "share_T"] if shap_df is not None else []) + ["raw_text"]
    wrong.sort_values("wrong_conf", ascending=False)[cols].head(40).to_csv(OUT / "tables" / f"error_cases_{tag}.csv", index=False)
    # 错误模式计数
    pat = wrong.groupby(["true_class", "pred_class"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    pat.to_csv(OUT / "tables" / f"error_patterns_{tag}.csv", index=False)
    return tab, df, cm, pat


def efficiency(preds, ds):
    rows = []
    if "TorchPredictor" not in globals() or not hasattr(next(iter(preds.values())), "state"):
        return pd.DataFrame()
    for key in [("base", CFG["DEPLOY_SEED"]), ("mg", CFG["DEPLOY_SEED"])]:
        if key not in preds:
            continue
        p0 = preds[key]
        p = TorchPredictor(p0.state, p0.gate_mode, key=p0.key)
        p._model()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t = time.time(); p.predict(ds, np.arange(ds.n))
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t_full = time.time() - t
        nm, cb, r, pos, ms = PANELS[ds.name]["PM"][-1]
        t = time.time(); p.predict(ds, np.arange(ds.n), ms)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        rows.append(dict(config=key[0], n=ds.n, ms_per_sample_complete=1000 * t_full / ds.n,
                         ms_per_sample_missing_TAV=1000 * (time.time() - t) / ds.n))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "efficiency.csv", index=False)
    return df


# %% [att3] 10. 附件3 推理（部署配置按验证集规则选定，部署种子事先固定）
def _apply_rules(rules, probs, yhat, indF=None):
    if rules is None:
        return probs.argmax(1), yhat, yhat
    try:
        return rules.apply(probs, yhat, indF)
    except TypeError:
        return rules.apply(probs, yhat)


def clean_sample_id(x):
    """附件3 的文件里带了 test 键，读出来的样本号形如 3_01_test；提交时去掉这个后缀。"""
    return re.sub(r"_(test|valid|train)$", "", str(x))


def run_att3(pred, rules, ds, deploy_name, members=(), member_rules=None, base_pred=None, base_rules=None, conf=None):
    """pred：部署预测器（单模型或集成）；members：[(名称, 预测器)]，用来算成员一致率；base_pred：对照的基础模型（或其集成）。"""
    full = pred.predict(ds, np.arange(ds.n))
    shap_df, _ = exact_shapley(pred, ds)
    phi, indT, indF = native_indicators(ds)
    c, y_out, y_cal = _apply_rules(rules, full["probs"], full["yhat"], indF)
    flag = review_flags(full, shap_df)
    votes = []
    for nm, p in members:
        f = p.predict(ds, np.arange(ds.n))
        rl = (member_rules or {}).get(nm, rules)
        votes.append(_apply_rules(rl, f["probs"], f["yhat"], indF)[0])
    votes = np.stack(votes, 1) if votes else c[:, None]
    agree = (votes == c[:, None]).mean(1)
    sid = [clean_sample_id(x) for x in ds.ids]
    sub = pd.DataFrame(dict(sample_id=sid, polarity=[CLASS_NAMES[k] for k in c], polarity_code=c - 1,
                            intensity=np.round(y_out, 4), p_negative=full["probs"][:, 0], p_neutral=full["probs"][:, 1],
                            p_positive=full["probs"][:, 2], confidence=full["probs"].max(1),
                            review_flag=flag.astype(int)))
    sub.to_csv(OUT / "attachment3_predictions.csv", index=False, encoding="utf-8-sig")
    det = sub.copy()
    det["source_id"] = ds.ids
    det["intensity_model_raw"] = full["yhat"]; det["intensity_calibrated"] = y_cal
    det["gate_text"] = full["gT"]; det["gate_fusion"] = full["gF"]
    for j, nm in enumerate(["miss_frac_T", "longest_gap_T", "has_miss_T", "miss_frac_A", "longest_gap_A", "has_miss_A",
                            "miss_frac_V", "longest_gap_V", "has_miss_V"]):
        det[nm] = phi[:, j]
    for m in MODS:
        det[f"share_cls_{m}"] = shap_df[f"share_cls_{m}"].values; det[f"phi_reg_{m}"] = shap_df[f"phi_reg_{m}"].values
    det["dominant_modality"] = shap_df.dom_cls.values
    det["member_agreement"] = agree; det["n_members"] = votes.shape[1]
    if base_pred is not None:
        base = base_pred.predict(ds, np.arange(ds.n))
        cb, yb, _ = _apply_rules(base_rules if base_rules is not None else rules, base["probs"], base["yhat"], indF)
        det["base_polarity"] = [CLASS_NAMES[k] for k in cb]; det["base_intensity"] = yb
        det["changed_vs_base"] = (cb != c).astype(int)
    det["deploy_config"] = deploy_name
    if conf is not None:
        for a in CFG["CONF_ALPHA"]:
            lo, hi, sets, sb = conf.apply(full["probs"], y_out, phi, a)
            det[f"intensity_lo_{int(100 * (1 - a))}"] = lo; det[f"intensity_hi_{int(100 * (1 - a))}"] = hi
            det[f"pred_set_{int(100 * (1 - a))}"] = sets
        det["severity_bin"] = sb
        det["review_flag_conformal"] = np.array([len(s.split("|")) > 1 for s in det[f"pred_set_{int(100 * (1 - CFG['CONF_ALPHA'][0]))}"]]).astype(int)
    det["words"] = ds.W.sum(1); det["raw_text"] = ds.raw_text
    det.to_csv(OUT / "attachment3_predictions_detail.csv", index=False, encoding="utf-8-sig")
    summ = dict(n=ds.n, deploy=deploy_name, polarity_counts=sub.polarity.value_counts().to_dict(),
                intensity_mean=float(y_out.mean()), confidence_mean=float(sub.confidence.mean()),
                review_flags=int(flag.sum()), dominant_counts=det.dominant_modality.value_counts().to_dict(),
                member_agreement_mean=float(agree.mean()),
                changed_vs_base=int(det["changed_vs_base"].sum()) if "changed_vs_base" in det else None,
                has_miss=dict(T=int(phi[:, 2].sum()), A=int(phi[:, 5].sum()), V=int(phi[:, 8].sum())),
                miss_frac_mean=dict(T=float(phi[:, 0].mean()), A=float(phi[:, 3].mean()), V=float(phi[:, 6].mean())),
                id_rename_example=f"{ds.ids[0]} → {sid[0]}" if ds.n else None)
    save_json(summ, OUT / "tables" / "att3_summary.json")
    log("附件3 汇总：", summ)
    return det, summ


# %% [att4] 11. 附件4：证据定位到原始视频（词位 → 秒 → 帧）
_MMS = {}


def match_videos(ds, vids):
    by_stem = {}
    for p in vids:
        by_stem.setdefault(_stem(p).lower(), []).append(p)
    out = {}
    for i, sid in enumerate(ds.ids):
        s = sid.lower()
        parts = s.split("$_$")
        vid, clip = (parts[0], parts[1]) if len(parts) == 2 else ("", "")
        cands = []
        for key in (s, s.replace("$_$", "_"), s.replace("$_$", "-"), clip):
            if key and key in by_stem:
                cands += by_stem[key]
        if vid and cands:
            cands = [p for p in cands if vid in p.lower()] or cands
        if not cands:
            nums = re.findall(r"\d+", s)
            if nums:
                k = nums[-1].lstrip("0") or "0"
                hit = [p for p in vids if (re.findall(r"\d+", _stem(p)) or ["x"])[-1].lstrip("0") == k]
                if len(hit) == 1:
                    cands = hit
        if cands:
            out[i] = (cands[0], "name")
    if len(out) < ds.n and len(vids) == ds.n:
        for i in range(ds.n):
            if i not in out:
                out[i] = (vids[i], "order")
    return out


def video_info(p):
    import cv2
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0; nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(fps), int(nfr), (float(nfr / fps) if fps > 0 else None)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _norm_word(w):
    return re.sub(r"[^a-z']", "", w.lower())


def mms_align(video, words):
    """MMS 强制对齐（torchaudio 自带，需联网下载权重）：已知词序列 → 每个词的起止秒。"""
    if _MMS.get("failed"):
        raise RuntimeError("MMS 模型不可用（此前加载失败），跳过")
    if "model" not in _MMS:
        try:
            import torchaudio
            b = torchaudio.pipelines.MMS_FA
            _MMS.update(model=b.get_model(with_star=False).to(DEVICE).eval(), tok=b.get_tokenizer(),
                        aligner=b.get_aligner(), sr=b.sample_rate)
            log("   MMS 强制对齐模型已加载")
        except Exception:
            _MMS["failed"] = True
            raise
    import soundfile as sf
    ff = _ffmpeg()
    if ff is None:
        raise RuntimeError("没有 ffmpeg")
    wav = OUT / "att4_keyframes" / "_tmp.wav"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", video, "-ac", "1", "-ar", "16000", str(wav)], check=True)
    x, sr = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == _MMS["sr"], f"采样率 {sr}"
    xt = torch.tensor(x)[None].to(DEVICE)
    with torch.inference_mode():
        em, _ = _MMS["model"](xt)
    em0 = em[0].float().cpu()
    spans = _MMS["aligner"](em0, _MMS["tok"](words))
    ratio = xt.shape[1] / em.shape[1] / sr
    out = [(float(s[0].start * ratio), float(s[-1].end * ratio)) for s in spans]
    # 对齐置信度：路径上逐字符概率的平均（强制对齐无论文音是否匹配都会给出路径，所以要看得分）
    score = float(np.mean([np.mean([t.score for t in s]) for s in spans])) if spans else float("nan")
    null = []
    rng = np.random.default_rng(len(words))
    for _ in range(CFG["ALIGN_NULL_PERM"] if len(words) > 2 else 0):
        pw = list(rng.permutation(words))
        try:
            sp2 = _MMS["aligner"](em0, _MMS["tok"](pw))
            null.append(float(np.mean([np.mean([t.score for t in s]) for s in sp2])))
        except Exception:
            pass
    dur = xt.shape[1] / sr
    _MMS["last_quality"] = dict(align_score=score, align_null=float(np.mean(null)) if null else float("nan"),
                                align_span_share=float((out[-1][1] - out[0][0]) / dur) if out and dur > 0 else float("nan"))
    return out


def att4_timing(ds, tokenizer):
    """每个样本：词片段（子词合并后的整词）→ 起止秒。优先 MMS 强制对齐；否则按词序在时长内均分（标注为近似）。"""
    vids = find_videos4()
    vm = match_videos(ds, vids)
    log(f"附件4 视频：找到 {len(vids)} 个，匹配 {len(vm)}/{ds.n}")
    ext = None
    if CFG["ATT4_ALIGN_CSV"] and os.path.exists(CFG["ATT4_ALIGN_CSV"]):
        ext = pd.read_csv(CFG["ATT4_ALIGN_CSV"])
    out = {}
    for i in range(ds.n):
        words, toks, spans = words_of(ds, i, tokenizer)
        span_words = [words[sp[0]] for sp in spans]
        info = dict(video=None, match=None, fps=None, n_frames=None, duration=None, method="none", pos_time={},
                    align_score=np.nan, align_null=np.nan, align_span_share=np.nan)
        if i in vm:
            info["video"], info["match"] = vm[i]
            try:
                info["fps"], info["n_frames"], info["duration"] = video_info(info["video"])
            except Exception as e:
                log("   读视频失败", info["video"], e)
        times = None
        if ext is not None:
            sub = ext[ext["sample"].astype(str) == ds.ids[i]]
            if len(sub):
                info["method"] = "external_csv"
                for _, r in sub.iterrows():
                    info["pos_time"][int(r.position)] = (float(r.start_s), float(r.end_s))
        if not info["pos_time"] and info["video"] and CFG["ATT4_TIMING"] == "auto":
            try:
                norm = [_norm_word(w) for w in span_words]
                keep = [j for j, w in enumerate(norm) if w]
                _MMS.pop("last_quality", None)
                al = mms_align(info["video"], [norm[j] for j in keep])
                times = [None] * len(spans)
                for j, t in zip(keep, al):
                    times[j] = t
                info["method"] = "mms_forced_alignment"
                info.update(_MMS.get("last_quality", {}))
            except Exception as e:
                if not _MMS.get("warned"):
                    log(f"   {ds.ids[i]}: MMS 强制对齐失败，改用按词位比例近似：", repr(e)[:160])
                    _MMS["warned"] = _MMS.get("failed", False)
                times = None
        if not info["pos_time"] and times is None and info["duration"] and CFG["ATT4_TIMING"] != "off":
            K = max(1, len(spans)); dur = info["duration"]
            times = [(dur * j / K, dur * (j + 1) / K) for j in range(K)]
            info["method"] = "proportional_approx"
        if times is not None:
            # 标点等无法对齐的词：取前后词时间插值
            for j in range(len(times)):
                if times[j] is None:
                    prev = next((times[k] for k in range(j - 1, -1, -1) if times[k] is not None), None)
                    nxt = next((times[k] for k in range(j + 1, len(times)) if times[k] is not None), None)
                    a = prev[1] if prev else (nxt[0] if nxt else 0.0); b = nxt[0] if nxt else a
                    times[j] = (a, max(a, b))
            for sp, t in zip(spans, times):
                for p in sp:
                    info["pos_time"][int(p)] = t
        out[i] = info
    return out


def pos_to_time(info, positions):
    pt = info["pos_time"]
    if not pt:
        return None
    keys = np.array(sorted(pt))
    ts = []
    for p in positions:
        k = keys[np.argmin(np.abs(keys - p))]           # 非词位置（如 [SEP] 处的声学/视觉帧）取最近的词
        ts.append(pt[int(k)])
    return (min(t[0] for t in ts), max(t[1] for t in ts))


def grab_frame(video, t_sec, out_path):
    import cv2
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000)
    ok, fr = cap.read()
    fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    if w > 360:
        fr = cv2.resize(fr, (360, int(h * 360 / w)))
    cv2.imwrite(str(out_path), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return fidx


def locate_evidence(cards, ds, timing):
    """给解释卡里的每条证据补上：原文词、起止秒、视频帧号；视觉证据额外截一张关键帧。"""
    for card in cards:
        i = ds.ids.index(card["id"])
        info = timing.get(i, {}) if timing else {}
        card["video"] = os.path.basename(info["video"]) if info.get("video") else None
        card["timing_method"] = info.get("method", "none")
        fps = info.get("fps")
        for m in MODS:
            for k, e in enumerate(card["evidence"][m]):
                t = pos_to_time(info, e["positions"]) if info else None
                if t is not None:
                    e["start_s"], e["end_s"] = round(t[0], 3), round(t[1], 3)
                    if fps:
                        e["frame_start"], e["frame_end"] = int(t[0] * fps), int(t[1] * fps)
                    if k == 0 and m in ("V", card["dominant_cls"]) and info.get("video"):
                        # 视觉证据取得分最高位置对应时刻的一帧；主导模态的首条证据也截一帧作为上下文
                        P = card["local_importance"][m]["positions"]; S = card["local_importance"][m]["score"]
                        best = [p for p in e["positions"]]
                        bp = max(best, key=lambda p: S[P.index(p)] if p in P else -1)
                        tb = pos_to_time(info, [bp])
                        fn = OUT / "att4_keyframes" / f"{re.sub(r'[^0-9A-Za-z_-]', '_', card['id'])}_{m}.jpg"
                        try:
                            fidx = grab_frame(info["video"], 0.5 * (tb[0] + tb[1]), fn)
                            if fidx is not None:
                                e["keyframe_index"] = fidx; e["keyframe_file"] = fn.name
                        except Exception as ex:
                            log("   截帧失败", ex)
    return cards


# %% [q3run] 12. 问题三：验证集（性能 + 解释质量）与附件4（全量预测 + 解释）
def summarize_faith(faith):
    if faith is None or not len(faith):
        return pd.DataFrame()
    g = faith.groupby(["modality", "budget"])[["del_top", "del_rand", "del_low", "keep_top", "keep_rand", "keep_low"]].mean()
    g["del_ratio_top_vs_rand"] = g.del_top / g.del_rand.replace(0, np.nan)
    g["top_beats_rand_share"] = faith.assign(w=faith.del_top > faith.del_rand).groupby(["modality", "budget"]).w.mean()
    return g.reset_index()


def run_q3_valid(pred, ds, tokenizer, freeze=False, tag="", shap_df=None, full=None):
    if shap_df is None:
        shap_df, full = exact_shapley(pred, ds)
    shap_df.to_csv(OUT / "q3" / "valid_shapley_all.csv", index=False)
    eff = dict(eff_err_cls_max=float(shap_df.eff_err_cls.max()), eff_err_reg_max=float(shap_df.eff_err_reg.max()))
    dom = dict(dominant_counts=shap_df.dom_cls.value_counts().to_dict(),
               dominant_reg_counts=shap_df.dom_reg.value_counts().to_dict(),
               mean_share_cls={m: float(shap_df[f"share_cls_{m}"].mean()) for m in MODS},
               mean_share_reg={m: float(shap_df[f"share_reg_{m}"].mean()) for m in MODS})
    by_class = shap_df.assign(true_class=[CLASS_NAMES[k] for k in ds.c]).groupby("true_class")[
        [f"share_cls_{m}" for m in MODS] + [f"share_reg_{m}" for m in MODS]].mean()
    by_class.to_csv(OUT / "q3" / "valid_share_by_class.csv")
    flag, rev = review_rule_eval(full, shap_df, ds)
    idx = pick_explain_valid(ds, CFG["N_EXPLAIN_VALID"])
    cards, faith, stab, imps = explain_set(pred, ds, idx, tokenizer, np.random.default_rng(11), shap_df, freeze=freeze)
    faith.to_csv(OUT / "q3" / f"valid_faithfulness_raw{tag}.csv", index=False)
    fs = summarize_faith(faith); fs.to_csv(OUT / "q3" / f"valid_faithfulness_summary{tag}.csv", index=False)
    stab.to_csv(OUT / "q3" / f"valid_stability{tag}.csv", index=False)
    save_json(cards, OUT / "q3" / f"valid_explanation_cards{tag}.json")
    res = dict(efficiency_check=eff, dominance=dom, review_rule=rev,
               stability_spearman_mean=float(stab.rho.mean()) if len(stab) else None,
               stability_by_modality=stab.groupby("modality").rho.mean().to_dict() if len(stab) else {},
               n_explained=len(idx), freeze_gate=bool(freeze),
               top_beats_rand_by_modality=faith.assign(w=faith.del_top > faith.del_rand).groupby("modality").w.mean().to_dict()
               if len(faith) else {})
    save_json(res, OUT / "q3" / f"valid_q3_summary{tag}.json")
    log("问题三（验证集）：", res)
    return shap_df, full, cards, faith, stab, res, imps


def _skey(x):
    return re.sub(r"^0+(?=\d)", "", clean_sample_id(str(x)).strip())


def load_manual_review():
    """人工核验表（优先于工具时间戳）：{样本号: dict(status, evidence, shift_s, note)}。"""
    rows = list(CFG.get("ATT4_MANUAL") or [])
    p = CFG.get("ATT4_MANUAL_CSV")
    if p and os.path.exists(p):
        rows = pd.read_csv(p, dtype=str).fillna("").to_dict("records")
    out = {}
    for r in rows:
        sh = r.get("shift_s")
        try:
            sh = float(sh) if sh not in (None, "", "nan") else None
        except Exception:
            sh = None
        out[_skey(r["sample"])] = dict(status=str(r.get("status", "")).strip() or "verified",
                                       evidence=str(r.get("evidence", "all")).strip() or "all", shift_s=sh,
                                       note=str(r.get("note", "")))
    return out


def timing_status(info, manual, med_score):
    """每个样本的时间戳状态：rejected / revised / revised_pending / flagged_unverified / tool_unverified / none。"""
    if manual:
        st = manual["status"]
        if st == "revised" and manual.get("shift_s") is None:
            return "revised_pending"
        return st
    if info.get("method") != "mms_forced_alignment":
        return "approx_unverified" if info.get("method") == "proportional_approx" else "none"
    sc, nl = info.get("align_score", np.nan), info.get("align_null", np.nan)
    if np.isfinite(sc) and ((np.isfinite(nl) and sc - nl < 0.05) or (np.isfinite(med_score) and sc < 0.5 * med_score)):
        return "flagged_unverified"
    return "tool_unverified"


def run_q3_att4(pred, rules, ds, tokenizer, data, freeze=False):
    shap_df, full = exact_shapley(pred, ds)
    idx = np.arange(ds.n)
    cards, faith, stab, imps = explain_set(pred, ds, idx, tokenizer, np.random.default_rng(12), shap_df, freeze=freeze)
    timing = att4_timing(ds, tokenizer) if CFG["ATT4_TIMING"] != "off" else {}
    cards = locate_evidence(cards, ds, timing)
    manual = load_manual_review()
    scores = [timing[i].get("align_score", np.nan) for i in timing] if timing else []
    med_score = float(np.nanmedian(scores)) if len(scores) and np.isfinite(scores).any() else np.nan
    c, y_out, y_cal = _apply_rules(rules, full["probs"], full["yhat"], native_indicators(ds)[2])
    # 附件4 中能在附件2 找回的样本（只做一致性参考，不作为评价依据）
    # 先按样本号匹配，再按词元序列（[CLS]…[SEP]）逐位相同匹配
    lookup, by_tok = {}, {}
    for k in ("train", "valid", "test"):
        if k in data:
            d2 = data[k]
            for j, sid in enumerate(d2.ids):
                lookup[sid] = (k, float(d2.y[j]), sid)
                by_tok[tuple(d2.input_ids[j, : d2.sep[j] + 1].tolist())] = (k, float(d2.y[j]), sid)
    for j, sid in enumerate(ds.ids):
        if sid not in lookup:
            hit = by_tok.get(tuple(ds.input_ids[j, : ds.sep[j] + 1].tolist()))
            if hit:
                lookup[sid] = hit
    rows = []
    for card, ci, yi in zip(cards, c, y_out):
        card["pred_class"] = CLASS_NAMES[ci]; card["intensity"] = float(yi)
        ev = card["evidence"]
        def fmt(m, key):
            e = ev[m][0] if ev[m] else {}
            return e.get(key)
        i = ds.ids.index(card["id"])
        info = timing.get(i, {}) if timing else {}
        man = manual.get(_skey(card["id"]))
        tst = timing_status(info, man, med_score)
        scope = (man or {}).get("evidence", "all")
        sh = (man or {}).get("shift_s")

        def final_t(m, t):
            if t is None:
                return None
            if scope not in ("all", m):
                return t
            if tst == "rejected":
                return None
            if tst == "revised" and sh is not None:
                return (round(max(0.0, t[0] + sh), 3), round(max(0.0, t[1] + sh), 3))
            return t
        tt = (fmt("T", "start_s"), fmt("T", "end_s")) if fmt("T", "start_s") is not None else None
        ta = (fmt("A", "start_s"), fmt("A", "end_s")) if fmt("A", "start_s") is not None else None
        tv = (fmt("V", "start_s"), fmt("V", "end_s")) if fmt("V", "start_s") is not None else None
        fv = (fmt("V", "frame_start"), fmt("V", "frame_end")) if fmt("V", "frame_start") is not None else None
        card["timing_status"] = tst; card["manual_review"] = man
        card["align_quality"] = {k: info.get(k) for k in ("align_score", "align_null", "align_span_share")}
        rows.append(dict(
            sample_id=clean_sample_id(card["id"]), polarity=CLASS_NAMES[ci], intensity=round(float(yi), 4), confidence=card["p_pred"],
            main_modality=card["dominant_cls"], main_modality_intensity=card["dominant_reg"],
            contrib_T=card["share_cls"]["T"], contrib_A=card["share_cls"]["A"], contrib_V=card["share_cls"]["V"],
            shapley_cls_T=card["shapley_cls"]["T"], shapley_cls_A=card["shapley_cls"]["A"], shapley_cls_V=card["shapley_cls"]["V"],
            shapley_reg_T=card["shapley_reg"]["T"], shapley_reg_A=card["shapley_reg"]["A"], shapley_reg_V=card["shapley_reg"]["V"],
            interaction_cls_TA=float(shap_df.set_index("id").loc[card["id"], "I_cls_TA"]),
            interaction_cls_TV=float(shap_df.set_index("id").loc[card["id"], "I_cls_TV"]),
            interaction_cls_AV=float(shap_df.set_index("id").loc[card["id"], "I_cls_AV"]),
            dominance_margin=card["margin_cls"],
            text_evidence=" | ".join(e["words"] for e in ev["T"]), text_positions="; ".join(str(e["positions"]) for e in ev["T"]),
            text_time_s=final_t("T", tt), audio_positions=fmt("A", "positions"), audio_words=fmt("A", "words"),
            audio_time_s=final_t("A", ta), visual_positions=fmt("V", "positions"), visual_words=fmt("V", "words"),
            visual_time_s=final_t("V", tv),
            visual_frames=None if final_t("V", tv) is None else (fv if tst != "revised" or sh is None else
                                                                 tuple(int(round(x * (info.get("fps") or 0))) for x in final_t("V", tv))),
            visual_keyframe=None if tst == "rejected" and scope in ("all", "V") else fmt("V", "keyframe_file"),
            timing_status=tst, manual_note=(man or {}).get("note"),
            align_score=info.get("align_score"), align_null=info.get("align_null"), align_span_share=info.get("align_span_share"),
            text_time_s_tool=tt, audio_time_s_tool=ta, visual_time_s_tool=tv,
            timing_method=card.get("timing_method"), video=card.get("video"),
            in_att2=lookup.get(card["id"], (None, None, None))[0], att2_label=lookup.get(card["id"], (None, None, None))[1],
            att2_id=lookup.get(card["id"], (None, None, None))[2],
            raw_text=ds.raw_text[i]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "attachment4_predictions_explanations.csv", index=False, encoding="utf-8-sig")
    save_json(cards, OUT / "attachment4_explanations.json")
    faith.to_csv(OUT / "q3" / "att4_faithfulness_raw.csv", index=False)
    summarize_faith(faith).to_csv(OUT / "q3" / "att4_faithfulness_summary.csv", index=False)
    stab.to_csv(OUT / "q3" / "att4_stability.csv", index=False)
    shap_df.to_csv(OUT / "q3" / "att4_shapley.csv", index=False)
    summ = dict(n=ds.n, polarity_counts=df.polarity.value_counts().to_dict(),
                main_modality_counts=df.main_modality.value_counts().to_dict(),
                mean_contrib={m: float(df[f"contrib_{m}"].mean()) for m in MODS},
                timing_methods=df.timing_method.value_counts().to_dict(),
                timing_status=df.timing_status.value_counts().to_dict(),
                align_score_median=med_score,
                n_keyframes=int(df.visual_keyframe.notna().sum()),
                recoverable_in_att2=int(df.in_att2.notna().sum()),
                stability_spearman_mean=float(stab.rho.mean()) if len(stab) else None,
                efficiency_err_max=float(shap_df.eff_err_cls.max()))
    rec = df[df.in_att2.notna()]
    if len(rec):
        cls_true = np.where(rec.att2_label < 0, 0, np.where(rec.att2_label > 0, 2, 1))
        summ["recoverable_polarity_match"] = int((cls_true == np.array([CLASS_NAMES.index(p) for p in rec.polarity])).sum())
        summ["recoverable_mae"] = float(np.abs(rec.att2_label - rec.intensity).mean())
        summ["recoverable_splits"] = rec.in_att2.value_counts().to_dict()
        summ["recoverable_note"] = ("能在附件2 找回的样本全部来自其测试集，只作重叠案例检查，不是额外的独立外部验证"
                                    if set(rec.in_att2) <= {"test"} else "找回样本的来源见 recoverable_splits")
    save_json(summ, OUT / "q3" / "att4_summary.json")
    log("附件4 汇总：", summ)
    return df, cards, shap_df, imps, summ


# %% [test] 13. 测试集一次性评估（所有选择都已在验证集上完成）
def run_test(preds, rules, deploy_seed, te):
    res, df, agg = evaluate_all(preds, te)
    boot = bootstrap_comparisons(res, te, deploy_seed)
    sdiff = seed_level_diffs(df, "test")
    rows = []
    for cfg_name in ("base", "mg"):
        for (cn, s), r in res.items():
            if cn != cfg_name:
                continue
            f = r["full"]
            m0 = r["complete"]
            row = dict(config=cn, seed=s, complete_f1=m0["f1"], complete_acc=m0["acc"], complete_mae=m0["mae"], complete_corr=m0["corr"])
            if rules is not None:
                c, y, _ = rules.apply(f["probs"], f["yhat"])
                m1 = metrics(te.c, te.y, f["probs"], y, c_pred=c)
                row.update(rules_f1=m1["f1"], rules_acc=m1["acc"], rules_mae=m1["mae"], rules_corr=m1["corr"])
                pm = []
                for (_, _, _, _, p) in r["PM"]:
                    c2, y2, _ = rules.apply(p["probs"], p["yhat"])
                    pm.append(metrics(te.c, te.y, p["probs"], y2, c_pred=c2))
                row.update(PM_avg_f1=r["PM_sum"]["avg_f1"], PM_avg_mae=r["PM_sum"]["avg_mae"],
                           rules_PM_avg_f1=float(np.mean([m["f1"] for m in pm])), rules_PM_avg_mae=float(np.mean([m["mae"] for m in pm])))
            rows.append(row)
    rt = pd.DataFrame(rows)
    rt.to_csv(OUT / "tables" / "test_rules_effect.csv", index=False)
    return res, df, agg, boot, sdiff, rt

# %% [v2meta] 16. v2 · 面板上的缺失指示与严重度（所有配置共用，只算一次）
def guard(name, fn, *a, **k):
    """非核心步骤出错时记录并跳过，保证后面的附件结果、报告和打包照常完成。"""
    try:
        return fn(*a, **k)
    except Exception as e:
        import traceback
        log(f"!!! {name} 失败：{e!r}（已跳过，其余步骤继续）")
        traceback.print_exc()
        S2.setdefault("errors", []).append(f"{name}: {e!r}")
        return None


_META = {}


def panel_meta(ds, kind):
    key = (ds.uid, kind)
    if key not in _META:
        out = []
        for nm, cb, r, pos, ms in PANELS[ds.name][kind]:
            aT, aA, aV, _ = apply_masks(ds, np.arange(ds.n), ms)
            phi, _, indF = descriptors(aT, aA, aV, ref_rows(ds, np.arange(ds.n)))
            out.append((indF, phi))
        _META[key] = out
    return _META[key]


def full_meta(ds):
    key = (ds.uid, "full")
    if key not in _META:
        phi, _, indF = native_indicators(ds)
        _META[key] = (indF, phi)
    return _META[key]


KINDS4 = ("PM", "LH", "SC", "MS")


def _gather(res, ds):
    full = res["full"]; indF_full, phi_full = full_meta(ds)
    Pb = {k: np.stack([x[4]["probs"] for x in res[k]]) for k in KINDS4}
    Yb = {k: np.stack([x[4]["yhat"] for x in res[k]]) for k in KINDS4}
    Ib = {k: np.stack([m[0] for m in panel_meta(ds, k)]) for k in KINDS4}
    Fb = {k: np.stack([m[1] for m in panel_meta(ds, k)]) for k in KINDS4}
    return full, indF_full, phi_full, Pb, Yb, Ib, Fb


# %% [v2dec] 17. v2 · 缺失感知的贝叶斯决策层（代价敏感分类的贝叶斯规则 ĉ = argmax_k w_k·p_k）
def _dec_grid():
    a, b = CFG["DEC_W_NEU"], CFG["DEC_W_POS"]
    wn = np.exp(np.linspace(np.log(a[0]), np.log(a[1]), int(a[2])))
    wp = np.exp(np.linspace(np.log(b[0]), np.log(b[1]), int(b[2])))
    return np.array([(1.0, x, y) for x in wn for y in wp])


def _conf_stack(c_true, pred):
    """pred: (B, n) → (B, 3, 3) 混淆矩阵（行=真实，列=预测）。"""
    B = pred.shape[0]
    idx = (np.arange(B)[:, None] * 9 + c_true[None, :] * 3 + pred).ravel()
    return np.bincount(idx, minlength=B * 9).reshape(B, 3, 3).astype(np.float64)


def _bucket_f1(C):
    tp = np.einsum("bii->bi", C); fp = C.sum(1) - tp; fn = C.sum(2) - tp
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0).mean(1)


def fit_w_complete(P, c, rows, grid):
    if rows.sum() == 0:
        return np.ones(3)
    Pm, cm = P[rows], c[rows]
    best, bw = -1.0, np.ones(3)
    for w in grid:
        f = _bucket_f1(_conf_stack(cm, np.argmax(Pm * w, 1)[None]))[0]
        if f > best + 1e-12:
            best, bw = f, w
    return bw


def fit_w_missing(Pb, c, Ib, w_c, grid):
    """各缺失档宏 F1 的平均最大；无缺失的行固定用 w_c。"""
    pred_c = np.argmax(Pb * w_c, -1)
    best, bw = -1.0, np.ones(3)
    for w in grid:
        pred = np.where(Ib > 0, np.argmax(Pb * w, -1), pred_c)
        f = _bucket_f1(_conf_stack(c, pred)).mean()
        if f > best + 1e-12:
            best, bw = f, w
    return bw


def _panel_summary(c_true, y_true, C, Y):
    """C, Y: (B, n)。返回平均宏 F1、最差档、平均 MAE、平均中性召回。"""
    Cm = _conf_stack(c_true, C)
    f1 = _bucket_f1(Cm)
    rec_neu = Cm[:, 1, 1] / np.maximum(Cm[:, 1, :].sum(1), 1)
    mae = np.abs(Y - y_true[None]).mean(1)
    return dict(avg_f1=float(f1.mean()), worst_f1=float(f1.min()), avg_mae=float(mae.mean()), rec_neu=float(rec_neu.mean()))


class RuleLayerV2:
    """缺失感知的贝叶斯决策层：完整输入用权重 w_c、有缺失用 w_m，ĉ = argmax_k w_k p_k；
    强度线性校准；可选极性—强度一致化。是否启用由验证集按视频分组 5 折的折外结果决定。"""

    def __init__(self, eps):
        self.eps = eps; self.w_c = np.ones(3); self.w_m = np.ones(3); self.calib = None; self.coherent = False
        self.use_dec = False; self.info = {}

    def classes(self, probs, indF=None):
        indF = np.zeros(len(probs)) if indF is None else np.asarray(indF)
        W = np.where(indF[:, None] > 0, self.w_m, self.w_c)
        return np.argmax(probs * W, 1)

    def apply(self, probs, yhat, indF=None):
        c = self.classes(probs, indF)
        ycal = apply_calib(yhat, self.calib)
        y = coherent_intensity(c, ycal, self.eps) if self.coherent else ycal
        return c, y, ycal


def fit_rules_v2(res, ds, eps, grid):
    full, indF_full, phi_full, Pb, Yb, Ib, Fb = _gather(res, ds)
    fitk = CFG["DEC_FIT_PANELS"]
    P_fit = np.concatenate([Pb[k] for k in fitk]); I_fit = np.concatenate([Ib[k] for k in fitk])
    c, y = ds.c, ds.y
    vids = np.array(ds.videos); uv = np.unique(vids)
    rng = np.random.default_rng(2026); rng.shuffle(uv)
    folds = np.array_split(uv, CFG["RULE_CV_FOLDS"])
    oc = {"full": np.zeros(ds.n, int), **{k: np.zeros(Pb[k].shape[:2], int) for k in KINDS4}}
    oy = {"full": np.zeros(ds.n), **{k: np.zeros(Pb[k].shape[:2]) for k in KINDS4}}
    for fv in folds:
        te = np.isin(vids, fv); tr = ~te
        w_c = fit_w_complete(full["probs"], c, tr & (indF_full == 0), grid)
        w_m = fit_w_missing(P_fit[:, tr], c[tr], I_fit[:, tr], w_c, grid)
        ab = fit_calib(full["yhat"][tr], y[tr])
        W = np.where(indF_full[te, None] > 0, w_m, w_c)
        oc["full"][te] = np.argmax(full["probs"][te] * W, 1)
        oy["full"][te] = apply_calib(full["yhat"][te], ab)
        for k in KINDS4:
            Wk = np.where(Ib[k][:, te, None] > 0, w_m, w_c)
            oc[k][:, te] = np.argmax(Pb[k][:, te] * Wk, -1)
            oy[k][:, te] = apply_calib(Yb[k][:, te], ab)
    raw_c = {"full": full["probs"].argmax(1), **{k: Pb[k].argmax(-1) for k in KINDS4}}
    raw_y = {"full": full["yhat"], **{k: Yb[k] for k in KINDS4}}

    def variant(cs, ys, coherent):
        out = {}
        yy = coherent_intensity(cs["full"], ys["full"], eps) if coherent else ys["full"]
        m = metrics(c, y, full["probs"], yy, c_pred=cs["full"])
        out.update(complete_f1=m["f1"], complete_acc=m["acc"], complete_mae=m["mae"], complete_corr=m["corr"],
                   complete_rec_neu=m["rec_neu"])
        for k in KINDS4:
            Y = np.stack([coherent_intensity(cs[k][b], ys[k][b], eps) for b in range(len(cs[k]))]) if coherent else ys[k]
            s = _panel_summary(c, y, cs[k], Y)
            out.update({f"{k}_{n}": v for n, v in s.items()})
        out["J"] = 0.5 * out["complete_f1"] + 0.5 * np.mean([out[f"{k}_avg_f1"] for k in fitk])
        return out

    V = {"raw": variant(raw_c, raw_y, False), "calib": variant(raw_c, oy, False),
         "calib+coherent": variant(raw_c, oy, True), "dec": variant(oc, raw_y, False),
         "dec+calib": variant(oc, oy, False), "dec+calib+coherent": variant(oc, oy, True)}
    use_dec = V["dec"]["J"] >= V["raw"]["J"] + CFG["RULE_MIN_GAIN"]
    b0 = "dec" if use_dec else "raw"
    b1 = ("dec+calib" if use_dec else "calib")
    use_cal = (V[b1]["complete_mae"] < V[b0]["complete_mae"] - 1e-4) and not (V[b1]["PM_avg_mae"] > V[b0]["PM_avg_mae"] + 0.002)
    b2 = b1 if use_cal else b0
    coh_name = ("dec+calib+coherent" if use_dec else "calib+coherent") if use_cal else None
    use_coh = coh_name is not None and V[coh_name]["complete_mae"] <= V[b2]["complete_mae"] + CFG["COHERENT_MAX_MAE_LOSS"]
    chosen = coh_name if use_coh else b2
    rl = RuleLayerV2(eps)
    rl.use_dec = bool(use_dec)
    if use_dec:
        rl.w_c = fit_w_complete(full["probs"], c, indF_full == 0, grid)
        rl.w_m = fit_w_missing(P_fit, c, I_fit, rl.w_c, grid)
    if use_cal:
        rl.calib = fit_calib(full["yhat"], y)
    rl.coherent = bool(use_coh)
    rl.info = dict(use_dec=bool(use_dec), use_calib=bool(use_cal), use_coherent=bool(use_coh), chosen=chosen,
                   w_complete=np.round(rl.w_c, 4).tolist(), w_missing=np.round(rl.w_m, 4).tolist(), calib=rl.calib,
                   cv={k: {n: float(v) for n, v in d.items()} for k, d in V.items()})
    return rl


def apply_rules_split(rl, res, ds):
    """把拟合好的规则层用到某个划分的完整输入与四套面板上，返回汇总指标。"""
    full, indF_full, phi_full, Pb, Yb, Ib, Fb = _gather(res, ds)
    c0, y0, _ = rl.apply(full["probs"], full["yhat"], indF_full)
    m = metrics(ds.c, ds.y, full["probs"], y0, c_pred=c0)
    out = dict(complete_f1=m["f1"], complete_acc=m["acc"], complete_mae=m["mae"], complete_corr=m["corr"],
               complete_rec_neu=m["rec_neu"])
    raw = metrics(ds.c, ds.y, full["probs"], full["yhat"])
    out.update(raw_complete_f1=raw["f1"], raw_complete_mae=raw["mae"], raw_complete_rec_neu=raw["rec_neu"])
    for k in KINDS4:
        C = np.stack([rl.apply(Pb[k][b], Yb[k][b], Ib[k][b])[0] for b in range(len(Pb[k]))])
        Y = np.stack([rl.apply(Pb[k][b], Yb[k][b], Ib[k][b])[1] for b in range(len(Pb[k]))])
        out.update({f"{k}_{n}": v for n, v in _panel_summary(ds.c, ds.y, C, Y).items()})
        out.update({f"raw_{k}_{n}": v for n, v in _panel_summary(ds.c, ds.y, Pb[k].argmax(-1), Yb[k]).items()})
    out["J"] = 0.5 * out["complete_f1"] + 0.5 * np.mean([out[f"{k}_avg_f1"] for k in CFG["DEC_FIT_PANELS"]])
    out["raw_J"] = 0.5 * out["raw_complete_f1"] + 0.5 * np.mean([out[f"raw_{k}_avg_f1"] for k in CFG["DEC_FIT_PANELS"]])
    return out


def rules_all_models(res_by_split, keys, eps, fit="valid_sel", report=("valid_cal", "test")):
    """v3：每个模型（配置×种子，或一个集成）各自在选择区 VS 上拟合规则层（VS 内按视频分组 5 折决定是否启用），
    再一次性应用到校准区 VC 与测试集。列名前缀：vs_cv_（VS 折外）、vc_、test_；raw 为 argmax 未加规则，rules 为加规则后。"""
    grid = _dec_grid()
    rules, rows = {}, []
    for key in keys:
        if key not in res_by_split[fit]:
            continue
        rl = fit_rules_v2(res_by_split[fit][key], DSETS[fit], eps, grid)
        rules[key] = rl
        cv = rl.info["cv"]; ch = cv[rl.info["chosen"]]
        row = dict(config=key[0], seed=key[1], chosen=rl.info["chosen"], use_dec=rl.info["use_dec"],
                   w_complete=rl.info["w_complete"], w_missing=rl.info["w_missing"], calib=rl.info["calib"])
        row.update({f"vs_cv_raw_{n}": v for n, v in cv["raw"].items()})
        row.update({f"vs_cv_rules_{n}": v for n, v in ch.items()})
        for sp in report:
            if sp in res_by_split and key in res_by_split[sp]:
                pre = {"valid_cal": "vc", "test": "test"}.get(sp, sp)
                te = apply_rules_split(rl, res_by_split[sp][key], DSETS[sp])
                row.update({f"{pre}_rules_{n}": v for n, v in te.items() if not n.startswith("raw_")})
                row.update({f"{pre}_raw_{n[4:]}": v for n, v in te.items() if n.startswith("raw_")})
        rows.append(row)
        log(f"  规则层 {key}: {rl.info['chosen']} | w_c={rl.info['w_complete']} w_m={rl.info['w_missing']} | "
            f"VS折外J {cv['raw']['J']:.4f}→{ch['J']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "v3_rules_all_models.csv", index=False)
    num = [c for c in df.columns if c.startswith(("vs_", "vc_", "test_")) and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("config")[num].agg(["mean", "std", "count"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg.to_csv(OUT / "tables" / "v3_rules_by_config.csv")
    return rules, df, agg


def rules_seed_diffs(df):
    from scipy.stats import t as tdist
    pairs = PAIRS
    mets = [c for c in df.columns if c.startswith(("vs_cv_rules_", "vc_rules_", "test_rules_")) and
            any(x in c for x in ("complete_f1", "complete_mae", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1",
                                 "MS_avg_f1", "PM_avg_mae", "_J"))]
    rows = []
    for a, b in pairs:
        A = df[df.config == a].set_index("seed"); B = df[df.config == b].set_index("seed")
        seeds = sorted(set(A.index) & set(B.index))
        if not seeds:
            continue
        for met in mets:
            d = (A.loc[seeds, met] - B.loc[seeds, met]).values.astype(float)
            k = len(d); mu = float(d.mean()); sd = float(d.std(ddof=1)) if k > 1 else float("nan")
            half = float(tdist.ppf(0.975, k - 1) * sd / math.sqrt(k)) if k > 1 else float("nan")
            rows.append(dict(compare=f"{a} - {b}", metric=met, n_seeds=k, mean=mu, lo=mu - half, hi=mu + half,
                             n_positive=int((d > 0).sum()), per_seed=[round(float(x), 5) for x in d]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tables" / "v3_rules_seed_diffs.csv", index=False)
    return out


def choose_deploy(df):
    """部署配置（事先定好的规则）：选择区 VS 折外 J（完整宏F1 与 PM/SC/MS 平均宏F1 各占一半，已含规则层）
    按种子平均最高者。单模型用事先固定的 DEPLOY_SEED；集成本身就是一个模型（种子记为 0）。测试集与 VC 不参与选择。"""
    s = df.groupby("config")["vs_cv_rules_J"].mean().sort_values(ascending=False)
    best = s.index[0]
    have = sorted(int(x) for x in df[df.config == best].seed)
    seed = 0 if best.endswith("_ens") else (CFG["DEPLOY_SEED"] if CFG["DEPLOY_SEED"] in have else have[0])
    info = dict(scores=s.round(5).to_dict(), deploy_config=best, deploy_seed=seed,
                rule="max mean VS-CV J over configs; singles deploy DEPLOY_SEED, ensembles seed 0")
    save_json(info, OUT / "tables" / "v3_deploy_choice.json")
    log("部署选择（选择区 VS 折外 J）：", info)
    return (best, seed), info


# %% [v2conf] 18. v2 · Mondrian 共形预测（按缺失严重度分桶的有限样本覆盖保证）
def sev_bin(phi):
    s = phi[:, [0, 3, 6]].max(1); e = CFG["SEV_EDGES"]
    b = np.digitize(s, e[1:-1], right=True) + 1
    return np.where(s <= 0, 0, b)


SEV_NAMES = {0: "complete", 1: "mild(0,0.2]", 2: "moderate(0.2,0.5]", 3: "severe(0.5,1]"}


class MondrianConformal:
    """强度：分数 |y − ŷ|，区间 ŷ ± q；极性：分数 1 − p_y，集合 {k : 1 − p_k ≤ q}。按缺失严重度分桶（Mondrian），桶内取
    ⌈(n+1)(1−α)⌉ 分位。v3 协议：校准区 VC 只做校准、不参与任何选择；每个样本每个桶至多一行（完整输入一行 + 随机一个缺失档一行，
    两行落在同一桶时只留完整输入那行），避免同一样本的多个缺失版本被当成独立校准点。
    保证的前提：VC 与测试样本可交换；样本按视频成簇，所以另外报告按视频重抽样的覆盖率区间。"""

    def __init__(self):
        self.q = {}

    def _rows(self, rl, res, ds, one_per_sample=True, seed=0):
        full, indF_full, phi_full, Pb, Yb, Ib, Fb = _gather(res, ds)
        rows_p, rows_yhat, rows_phi, rows_ind, rows_i = [full["probs"]], [full["yhat"]], [phi_full], [indF_full], [np.arange(ds.n)]
        allP = np.concatenate([Pb[k] for k in KINDS4]); allY = np.concatenate([Yb[k] for k in KINDS4])
        allI = np.concatenate([Ib[k] for k in KINDS4]); allF = np.concatenate([Fb[k] for k in KINDS4])
        if one_per_sample:
            # 每个样本、每个缺失桶至多一行：在该样本落入此桶的所有缺失档里等概率抽一个（与测试端“桶内全部档等权”一致）；
            # 与完整输入那行同桶的不再抽（原生缺失样本的完整输入可能已经在缺失桶里）
            rng = np.random.default_rng(seed)
            Bk = allF.shape[0]
            bins_all = sev_bin(allF.reshape(-1, allF.shape[-1])).reshape(Bk, ds.n)
            b0 = sev_bin(phi_full)
            bb, ii = [], []
            for i in range(ds.n):
                for b in (1, 2, 3):
                    if b == b0[i]:
                        continue
                    cand = np.flatnonzero(bins_all[:, i] == b)
                    if len(cand):
                        bb.append(int(rng.choice(cand))); ii.append(i)
            b = np.array(bb, int); i = np.array(ii, int)
            rows_p.append(allP[b, i]); rows_yhat.append(allY[b, i]); rows_phi.append(allF[b, i]); rows_ind.append(allI[b, i]); rows_i.append(i)
        else:
            for b in range(allP.shape[0]):
                rows_p.append(allP[b]); rows_yhat.append(allY[b]); rows_phi.append(allF[b]); rows_ind.append(allI[b]); rows_i.append(np.arange(ds.n))
        P = np.concatenate(rows_p); Yh = np.concatenate(rows_yhat); Phi = np.concatenate(rows_phi)
        Ind = np.concatenate(rows_ind); I = np.concatenate(rows_i)
        c, yout, _ = rl.apply(P, Yh, Ind) if rl is not None else (P.argmax(1), Yh, Yh)
        return P, yout, Phi, I

    def fit(self, rl, res, ds):
        P, yout, Phi, I = self._rows(rl, res, ds, one_per_sample=True)
        s_int = np.abs(ds.y[I] - yout); s_cls = 1 - P[np.arange(len(I)), ds.c[I]]
        bins = sev_bin(Phi)
        vids = np.array(ds.videos)
        self.n = {int(b): int((bins == b).sum()) for b in range(4)}
        self.n_videos = {int(b): int(len(set(vids[I[bins == b]]))) for b in range(4)}
        self.calib_split = ds.name
        for a in CFG["CONF_ALPHA"]:
            for b in range(4):
                m = bins == b; n = int(m.sum())
                k = math.ceil((n + 1) * (1 - a))
                qi = float(np.sort(s_int[m])[k - 1]) if 0 < k <= n else float("inf")
                qc = float(np.sort(s_cls[m])[k - 1]) if 0 < k <= n else 1.0
                self.q[(a, b)] = (qi, qc)
        return self

    def apply(self, probs, yout, phi, a):
        bins = sev_bin(phi)
        qi = np.array([self.q[(a, int(b))][0] for b in bins]); qc = np.array([self.q[(a, int(b))][1] for b in bins])
        lo = np.clip(yout - qi, -3, 3); hi = np.clip(yout + qi, -3, 3)
        sets = []
        for p, q in zip(probs, qc):
            ks = [k for k in range(3) if 1 - p[k] <= q] or [int(np.argmax(p))]
            sets.append("|".join(CLASS_NAMES[k] for k in ks))
        return lo, hi, np.array(sets), bins

    def evaluate(self, rl, res, ds, reps=None):
        """测试端：全部面板行都评估（同一样本的多个缺失版本高度相关），覆盖率区间按源视频重抽样得到。"""
        P, yout, Phi, I = self._rows(rl, res, ds, one_per_sample=False)
        uv, inv = np.unique(np.array(ds.videos), return_inverse=True)
        rv = inv[I]
        rng = np.random.default_rng(5); reps = reps or CFG["BOOT_REPS"]
        W = [np.bincount(rng.integers(0, len(uv), len(uv)), minlength=len(uv)).astype(np.float64) for _ in range(reps)]
        rows = []
        for a in CFG["CONF_ALPHA"]:
            lo, hi, sets, bins = self.apply(P, yout, Phi, a)
            cov_i = ((ds.y[I] >= lo - 1e-9) & (ds.y[I] <= hi + 1e-9)).astype(np.float64)
            size = np.array([len(s.split("|")) for s in sets])
            cov_c = np.array([CLASS_NAMES[ds.c[i]] in s.split("|") for i, s in zip(I, sets)]).astype(np.float64)
            for b in range(4):
                m = bins == b
                if not m.any():
                    continue
                bi = [np.sum(w[rv[m]] * cov_i[m]) / max(np.sum(w[rv[m]]), 1e-12) for w in W]
                bc = [np.sum(w[rv[m]] * cov_c[m]) / max(np.sum(w[rv[m]]), 1e-12) for w in W]
                comp = pd.Series(sets[m]).value_counts(normalize=True).round(3).to_dict()
                rows.append(dict(alpha=a, target=1 - a, bin=SEV_NAMES[b], n_rows=int(m.sum()),
                                 n_test_samples=int(len(set(I[m]))), n_test_videos=int(len(set(rv[m]))),
                                 calib_split=self.calib_split, n_calib=self.n.get(b), n_calib_videos=self.n_videos.get(b),
                                 q_int=self.q[(a, b)][0], coverage_int=float(cov_i[m].mean()),
                                 coverage_int_lo=float(np.percentile(bi, 2.5)), coverage_int_hi=float(np.percentile(bi, 97.5)),
                                 width=float((hi - lo)[m].mean()), q_cls=self.q[(a, b)][1], coverage_set=float(cov_c[m].mean()),
                                 coverage_set_lo=float(np.percentile(bc, 2.5)), coverage_set_hi=float(np.percentile(bc, 97.5)),
                                 avg_set_size=float(size[m].mean()), singleton=float((size[m] == 1).mean()),
                                 set_composition=json.dumps(comp, ensure_ascii=False)))
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "tables" / f"v3_conformal_{ds.name}.csv", index=False)
        return df


# %% [v2ext] 19. v2 · 多种子扩展扫描（缺失比例 0.1 → 1.0）
def ext_scan_all(preds, data):
    rows = []
    for sp in CFG["EXT_SPLITS"]:
        if sp not in data:
            continue
        ds = data[sp]
        panel = [x for x in build_panel(ds, "PM", ratios=CFG["EXT_RATIOS"], allow_clear=True) if x[1] in ("T", "A", "V", "AV", "TAV")]
        for cfg in CFG["EXT_CONFIGS"]:
            for s in CFG["SEEDS_MAIN"]:
                if (cfg, s) not in preds:
                    continue
                for nm, cb, r, pos, ms in panel:
                    p = preds[(cfg, s)].predict(ds, np.arange(ds.n), ms)
                    m = metrics(ds.c, ds.y, p["probs"], p["yhat"])
                    rows.append(dict(split=sp, config=cfg, seed=s, combo=cb, ratio=r, pos=pos, f1=m["f1"], acc=m["acc"],
                                     mae=m["mae"], rec_neu=m["rec_neu"], gateF=float(p["gF"].mean()), gateT=float(p["gT"].mean())))
            log(f"  扩展扫描 {sp} {cfg} 完成")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "v2_ext_scan_raw.csv", index=False)
    g = df.groupby(["split", "config", "seed", "combo", "ratio"])[["f1", "mae", "rec_neu", "gateF", "gateT"]].mean().reset_index()
    from scipy.stats import t as tdist
    diffs = []
    for (sp, cb, r), d in g.groupby(["split", "combo", "ratio"]):
        base = d[d.config == "base"].set_index("seed")
        for cfg in [c for c in CFG["EXT_CONFIGS"] if c != "base"]:
            x = d[d.config == cfg].set_index("seed")
            ss = sorted(set(base.index) & set(x.index))
            if len(ss) < 2:
                continue
            for met in ("f1", "mae"):
                v = (x.loc[ss, met] - base.loc[ss, met]).values
                half = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
                diffs.append(dict(split=sp, combo=cb, ratio=r, compare=f"{cfg} - base", metric=met, mean=v.mean(),
                                  lo=v.mean() - half, hi=v.mean() + half, n_positive=int((v > 0).sum()), n=len(v)))
    dd = pd.DataFrame(diffs)
    dd.to_csv(OUT / "tables" / "v2_ext_scan_diffs.csv", index=False)
    return df, g, dd


# %% [v2theory] 20. v2 · 命题 3 的数值检验（连续 vs 散点缺失）与低秩检验
def _mstd_np(x, a):
    a = a[..., None].astype(np.float32); c = np.maximum(a.sum(1), 1)
    mu = (x * a).sum(1) / c; var = (((x - mu[:, None]) ** 2) * a).sum(1) / c
    return np.concatenate([mu, np.sqrt(var + 1e-8)], 1)


def _masks_k(o_rows, k, mode, rng):
    ms = np.zeros_like(o_rows)
    for i in range(len(o_rows)):
        P = np.flatnonzero(o_rows[i]); kk = min(int(k[i]), len(P) - 1)
        if kk <= 0:
            continue
        if mode == "contig":
            st = int(rng.integers(0, len(P) - kk + 1)); ms[i, P[st:st + kk]] = True
        else:
            ms[i, rng.choice(P, kk, replace=False)] = True
    return ms


def _autocorr(X, o, maxlag):
    """逐滞后估计：滞后 k 用所有长度 ≥ k+3 的样本（v2 只用长度 ≥ maxlag+3 的样本，滞后取大时会丢掉短句）。"""
    rho = np.zeros(maxlag + 1); cnt = np.zeros(maxlag + 1)
    for i in range(len(X)):
        P = np.flatnonzero(o[i])
        if len(P) < 3:
            continue
        x = X[i, P].astype(np.float64)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        for k in range(min(maxlag, len(P) - 3) + 1):
            rho[k] += (x[: len(P) - k] * x[k:]).mean(); cnt[k] += 1
    return rho / np.maximum(cnt, 1)


def _theory_dev(L, sel, rho):
    """池化均值偏差的理论值 E‖μ̂_obs − μ̂‖²/σ² = aᵀΣa，Σ_ij = ρ(|i−j|)（超出估计阶数的滞后取 0）。"""
    lag = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    R = np.where(lag < len(rho), rho[np.minimum(lag, len(rho) - 1)], 0.0)
    a = np.full(L, -1.0 / L); keep = np.ones(L, bool); keep[sel] = False
    a[keep] += 1.0 / keep.sum()
    return float(a @ R @ a)


def prop3_check(pred_base, ds, maxlag=8, model_level=True):
    rng = np.random.default_rng(3)
    ok = (ds.oT.sum(1) >= 10) & (ds.oA.sum(1) >= 10) & (ds.oV.sum(1) >= 10)
    idx = np.flatnonzero(ok)[: CFG["REPR_N"]]
    full = pred_base.predict(ds, idx, hidden=True)
    tok = pred_base.token_states(ds, idx)
    feats = {"T": tok, "A": ds.audio[idx], "V": ds.vision[idx]}
    rhos = {m: _autocorr(feats[m], ds.o(m)[idx], maxlag) for m in MODS}
    z_full = {"T": full["zT"], "A": _mstd_np(ds.audio[idx], ds.oA[idx]), "V": _mstd_np(ds.vision[idx], ds.oV[idx])}
    rows = []
    for m in MODS:
        o = ds.o(m)[idx]; L = o.sum(1)
        X = feats[m]
        sd_all = np.stack([X[i, np.flatnonzero(o[i])].std(0) + 1e-6 for i in range(len(idx))])
        mu_full = np.stack([X[i, np.flatnonzero(o[i])].mean(0) for i in range(len(idx))])
        dz_den = ((z_full[m] - z_full[m].mean(0)) ** 2).sum(1).mean()
        dh_den = ((full["h"] - full["h"].mean(0)) ** 2).sum(1).mean()
        for r in CFG["REPR_RATIOS"]:
            k = np.maximum(1, np.round(r * L)).astype(int)
            for mode in ("contig", "scatter"):
                mm = _masks_k(o, k, mode, rng)
                # (1) 只看池化：固定逐位置表示，删位置后重新取均值（理论与实测同口径，单位 σ²）
                emp, th = [], []
                for i in range(len(idx)):
                    P = np.flatnonzero(o[i]); keep = o[i] & ~mm[i]
                    if keep.sum() == 0:
                        continue
                    emp.append(float((((X[i, keep].mean(0) - mu_full[i]) / sd_all[i]) ** 2).mean()))
                    sel = np.flatnonzero(np.isin(P, np.flatnonzero(mm[i])))
                    th.append(_theory_dev(len(P), sel, rhos[m]))
                row = dict(modality=m, ratio=r, mode=mode, maxlag=maxlag, pool_dev_empirical=float(np.mean(emp)),
                           pool_dev_theory=float(np.mean(th)), n_deleted_mean=float(mm.sum(1).mean()))
                # (2) 整个模型：表示与预测的实际变化（删除个数与连续/散点完全匹配，文本走同一条重新编码路径）
                if model_level:
                    ms = {x: (mm if x == m else np.zeros_like(mm)) for x in MODS}
                    p = pred_base.predict(ds, idx, ms, hidden=True)
                    z = p["zT"] if m == "T" else _mstd_np((ds.audio if m == "A" else ds.vision)[idx], o & ~mm)
                    row.update(repr_dev=float(((z - z_full[m]) ** 2).sum(1).mean() / dz_den),
                               fused_dev=float(((p["h"] - full["h"]) ** 2).sum(1).mean() / dh_den),
                               pred_tv=float(np.abs(p["probs"] - full["probs"]).sum(1).mean() / 2),
                               flip_rate=float((p["probs"].argmax(1) != full["probs"].argmax(1)).mean()))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"v3_prop3_check_lag{maxlag}.csv", index=False)
    rho_df = pd.DataFrame({m: rhos[m] for m in MODS}); rho_df.index.name = "lag"
    rho_df.to_csv(OUT / "tables" / f"v3_autocorr_lag{maxlag}.csv")
    return df, rho_df


def lowrank_check(pred_base, ds):
    """缺失引起的表示偏移是否集中在少数方向：对 Δh（融合隐层）与 ΔzT（文本池化）做特征值分解，
    与同形状高斯随机矩阵对比累计能量。"""
    idx = np.arange(min(ds.n, CFG["SVD_N"]))
    full = pred_base.predict(ds, idx, hidden=True)
    Dh, Dz = [], []
    for nm, cb, r, pos, ms in PANELS[ds.name]["PM"]:
        ms_i = {m: ms[m][idx] for m in MODS}
        p = pred_base.predict(ds, idx, ms_i, hidden=True)
        ch = np.abs(p["h"] - full["h"]).sum(1) > 1e-6
        Dh.append((p["h"] - full["h"])[ch])
        if "T" in cb:
            chz = np.abs(p["zT"] - full["zT"]).sum(1) > 1e-6
            Dz.append((p["zT"] - full["zT"])[chz])
    rows = []
    for name, D in (("fused_h", np.concatenate(Dh)), ("text_z", np.concatenate(Dz) if Dz else np.zeros((0, 1)))):
        if len(D) < 5:
            continue
        ev = np.sort(np.linalg.eigvalsh(D.T.astype(np.float64) @ D.astype(np.float64)))[::-1]
        ev = np.clip(ev, 0, None); e = np.cumsum(ev) / ev.sum()
        R = np.random.default_rng(0).normal(size=D.shape)
        er = np.sort(np.linalg.eigvalsh(R.T @ R))[::-1]; er = np.cumsum(er) / er.sum()
        pn = ev / ev.sum(); eff_rank = float(np.exp(-(pn[pn > 0] * np.log(pn[pn > 0])).sum()))
        for rk in (1, 2, 4, 8, 16, 32):
            if rk <= len(e):
                rows.append(dict(space=name, dim=D.shape[1], n_rows=len(D), rank=rk, energy=float(e[rk - 1]),
                                 energy_random=float(er[rk - 1]), effective_rank=eff_rank))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "v2_lowrank_check.csv", index=False)
    return df


def interactions_summary(shap_df, ds, tag):
    cols = [c for c in shap_df.columns if c.startswith("I_")]
    X = shap_df[cols].to_numpy(dtype=np.float64)          # 直接用 numpy，避开 pandas 2.x 对 lambda 聚合的限制
    s = pd.DataFrame({"mean": X.mean(0), "mean_abs": np.abs(X).mean(0), "share_positive": (X > 0).mean(0)}, index=cols)
    s.to_csv(OUT / "q3" / f"interactions_{tag}.csv")
    if ds.has_label:
        bc = shap_df.assign(true_class=[CLASS_NAMES[k] for k in ds.c]).groupby("true_class")[cols].mean()
        bc.to_csv(OUT / "q3" / f"interactions_by_class_{tag}.csv")
    return s


def gate_coef_table(preds):
    rows = []
    names_T = ["frac_T", "gap_T"]; names_F = ["frac_T", "gap_T", "frac_A", "gap_A", "frac_V", "gap_V"]
    for (cfg, s), p in preds.items():
        if cfg != "mg2" or not hasattr(p, "gate_coef"):
            continue
        gc = p.gate_coef()
        rows.append(dict(seed=s, gate="text", **{n: v for n, v in zip(names_T, gc["text"])}))
        rows.append(dict(seed=s, gate="fusion", **{n: v for n, v in zip(names_F, gc["fusion"])}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "v2_gate_coef.csv", index=False)
    return df

# %% [v3ens] 23. v3 · 集成（直接用成员已经算好的预测，不再重新推理）
def _family_combine(preds_by_family):
    """先在每个主干家族内平均，再在家族之间平均（BERT 与 DeBERTa 各占一半，不受成员个数影响）。"""
    fams = [combine_preds(v) for v in preds_by_family if v]
    return fams[0] if len(fams) == 1 else combine_preds(fams)


def _ens_res(res_split, fams, ds):
    keys = {f: sorted(k for k in res_split if k[0] == f) for f in fams}
    if not all(keys.values()):                 # 任一家族没有成员（如 DeBERTa 训练失败）就不构造这个集成
        return None, []
    r = dict(full=_family_combine([[res_split[k]["full"] for k in keys[f]] for f in fams]))
    for kind in PANEL_KINDS:
        ref = next(res_split[k][kind] for f in fams for k in keys[f])
        out = []
        for b, (nm, cb, rr, pos, _) in enumerate(ref):
            out.append((nm, cb, rr, pos, _family_combine([[res_split[k][kind][b][4] for k in keys[f]] for f in fams])))
        r[kind] = out
    if ds.has_label:
        r["complete"] = metrics(ds.c, ds.y, r["full"]["probs"], r["full"]["yhat"])
        for kind in PANEL_KINDS:
            df = panel_table(ds, r[kind]); r[kind + "_df"] = df
            r[kind + "_sum"] = panel_summary(df, r["complete"])
    return r, [k for f in fams for k in keys[f]]


def add_ensembles(res_split, sums_df, ds):
    rows = []
    for name, fams in CFG["ENSEMBLES"].items():
        r, members = _ens_res(res_split, fams, ds)
        if r is None:
            continue
        res_split[(name, 0)] = r
        if ds.has_label:
            row = dict(split=ds.name, config=name, seed=0, n_members=len(members))
            row.update({f"complete_{k}": v for k, v in r["complete"].items()})
            for kind in PANEL_KINDS:
                row.update({f"{kind}_{k}": v for k, v in r[kind + "_sum"].items()})
            rows.append(row)
            log(f"  {ds.name}_{name}（{len(members)} 个成员）: 完整F1={r['complete']['f1']:.4f} Acc={r['complete']['acc']:.4f} "
                f"MAE={r['complete']['mae']:.4f} | PM均F1={r['PM_sum']['avg_f1']:.4f}")
    if rows and sums_df is not None:
        sums_df = pd.concat([sums_df, pd.DataFrame(rows)], ignore_index=True)
        sums_df.to_csv(OUT / "tables" / f"summary_runs_{ds.name}.csv", index=False)
    return res_split, sums_df


def ensemble_predictors(preds):
    out = {}
    for name, fams in CFG["ENSEMBLES"].items():
        groups = [(f, [preds[k] for k in sorted(preds) if k[0] == f]) for f in fams]
        if not all(g for _, g in groups):
            continue
        fam_preds = [EnsemblePredictor(g, name=f"{f}_ens") if len(g) > 1 else g[0] for f, g in groups]
        out[(name, 0)] = fam_preds[0] if len(fam_preds) == 1 else EnsemblePredictor(fam_preds, name=name)
    return out


def config_summary(sums, split):
    df = sums[split]
    num = [c for c in df.columns if c not in ("split", "config", "seed", "n_members") and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("config")[num].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(OUT / "tables" / f"summary_by_config_{split}.csv")
    return agg


# %% [v3audit] 24. v3 · v2 结论补验（只推理）：90% 文本档逐样本、joint 对照、原生缺失子集
def _rem_words(ds, aT, tokenizer):
    """每行剩余的“含字母或数字的整词”个数（有至少一个子词位置保留就算保留）。"""
    out = np.zeros(ds.n, np.int64)
    for i in range(ds.n):
        words, _, spans = words_of(ds, i, tokenizer)
        out[i] = sum(1 for sp in spans if aT[i, sp].any() and _ALNUM.search(words[sp[0]] or ""))
    return out


REM_BINS = [(-1, 0, "0"), (0, 1, "1"), (1, 2, "2"), (2, 4, "3-4"), (4, 99, ">=5")]


def severe_text_audit(preds, ds, tokenizer, configs=None, ratios=None):
    """只删文本、比例 0.75/0.9/1.0（头/中/尾）；逐样本记录剩余子词数与剩余整词数、各模型的预测。
    检验两件事：① 收益是否随“残余语义”变化（按剩余整词数分层）；② 常规缺失增广训练 joint 是否也有同样的收益。"""
    configs = [c for c in (configs or CFG["AUDIT_CONFIGS"]) if any(k[0] == c for k in preds)]
    ratios = ratios or CFG["AUDIT_RATIOS"]
    panel = [x for x in build_panel(ds, "PM", ratios=ratios, allow_clear=True) if x[1] == "T"]
    rows = []
    for nm, cb, r, pos, ms in panel:
        aT = ds.oT & ~ms["T"]
        rem_pos = aT.sum(1); rem_w = _rem_words(ds, aT, tokenizer)
        base = dict(split=ds.name, ratio=r, pos=pos, id=ds.ids, video=ds.videos, y=ds.y, c=ds.c,
                    n_pos=ds.oT.sum(1), rem_pos=rem_pos, rem_words=rem_w)
        for cfg in configs:
            for key in sorted(k for k in preds if k[0] == cfg):
                p = preds[key].predict(ds, np.arange(ds.n), ms)
                d = pd.DataFrame(base); d["config"] = cfg; d["seed"] = key[1]
                d["c_pred"] = p["probs"].argmax(1); d["yhat"] = p["yhat"]; d["p_true"] = p["probs"][np.arange(ds.n), ds.c]
                d["gF"] = p["gF"]; d["gT"] = p["gT"]
                rows.append(d)
        log(f"  补验 {ds.name} T|{r}|{pos} 完成")
    df = pd.concat(rows, ignore_index=True)
    df["rem_bin"] = "?"
    for lo, hi, lab in REM_BINS:
        df.loc[(df.rem_words > lo) & (df.rem_words <= hi), "rem_bin"] = lab
    df[df.ratio < 1.0].to_csv(OUT / "tables" / f"v3_audit_severe_text_samples_{ds.name}.csv.gz", index=False, compression="gzip")
    return df


def _f1_acc(d):
    C = conf_mat(d.c.values, d.c_pred.values)
    return f1_from_cm(C)[0], float((d.c.values == d.c_pred.values).mean()), float(np.abs(d.y.values - d.yhat.values).mean())


def audit_strata(df, ds, reps=None):
    """每个 (比例, 剩余整词分层, 配置)：各种子的宏F1/Acc/MAE 取均值；与 base 的同种子配对差（t 区间）和按视频重抽样区间。"""
    from scipy.stats import t as tdist
    reps = reps or CFG["BOOT_REPS"]
    out = []
    strata = [("all", None)] + [(lab, lab) for _, _, lab in REM_BINS]
    uv, inv_all = np.unique(np.array(ds.videos), return_inverse=True)
    vid_of = dict(zip(ds.ids, inv_all))
    rng = np.random.default_rng(8)
    Wb = [np.bincount(rng.integers(0, len(uv), len(uv)), minlength=len(uv)).astype(np.float64) for _ in range(reps)]
    for r in sorted(df.ratio.unique()):
        dr = df[df.ratio == r]
        for sname, lab in strata:
            ds_ = dr if lab is None else dr[dr.rem_bin == lab]
            if not len(ds_):
                continue
            per = {}
            for (cfg, sd), g in ds_.groupby(["config", "seed"]):
                per[(cfg, sd)] = g
            base_seeds = sorted(sd for (cfg, sd) in per if cfg == "base")
            for cfg in sorted(set(c for c, _ in per)):
                seeds = sorted(sd for (c, sd) in per if c == cfg)
                vals = np.array([_f1_acc(per[(cfg, sd)]) for sd in seeds])
                row = dict(split=ds.name, ratio=r, stratum=sname, config=cfg, n_rows_per_seed=int(len(per[(cfg, seeds[0])])),
                           n_seeds=len(seeds), f1=float(vals[:, 0].mean()), acc=float(vals[:, 1].mean()), mae=float(vals[:, 2].mean()))
                common = [sd for sd in seeds if sd in base_seeds]
                if cfg != "base" and len(common) >= 1:
                    dv = np.array([np.array(_f1_acc(per[(cfg, sd)])) - np.array(_f1_acc(per[("base", sd)])) for sd in common])
                    for j, met in enumerate(("f1", "acc", "mae")):
                        mu = dv[:, j].mean()
                        half = (tdist.ppf(0.975, len(dv) - 1) * dv[:, j].std(ddof=1) / math.sqrt(len(dv))) if len(dv) >= 2 else np.nan
                        row[f"d_{met}"] = float(mu); row[f"d_{met}_seed_lo"] = float(mu - half); row[f"d_{met}_seed_hi"] = float(mu + half)
                    row["n_positive_f1"] = int((dv[:, 0] > 0).sum()); row["n_common_seeds"] = len(common)
                    # 按视频重抽样：同一批（样本×位置）行，种子平均后的差值
                    A = [per[(cfg, sd)] for sd in common]; B = [per[("base", sd)] for sd in common]
                    vA = np.array([vid_of[x] for x in A[0].id.values])
                    bs = []
                    for w in Wb:
                        ww = w[vA]
                        if ww.sum() == 0:
                            continue
                        fa = np.mean([f1_from_cm(conf_mat(a.c.values, a.c_pred.values, ww))[0] for a in A])
                        fb = np.mean([f1_from_cm(conf_mat(b.c.values, b.c_pred.values, ww))[0] for b in B])
                        aa = np.mean([np.sum(ww * (a.c.values == a.c_pred.values)) / ww.sum() for a in A])
                        ab = np.mean([np.sum(ww * (b.c.values == b.c_pred.values)) / ww.sum() for b in B])
                        bs.append((fa - fb, aa - ab))
                    bs = np.array(bs)
                    row["d_f1_video_lo"], row["d_f1_video_hi"] = (float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5)))
                    row["d_acc_video_lo"], row["d_acc_video_hi"] = (float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5)))
                out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(OUT / "tables" / f"v3_audit_severe_text_strata_{ds.name}.csv", index=False)
    return t


def native_missing_audit(res_split, ds, cfgs=("mg", "mg2", "joint")):
    """原生缺失子集（完整输入里本来就有缺失的样本）：门触发率、类别改变数、MAE/Acc 变化；无原生缺失子集：与 base 的最大差值。"""
    _, _, indF = native_indicators(ds)
    nat = indF > 0
    rows = []
    for (cfg, s), r in res_split.items():
        if cfg not in cfgs or ("base", s) not in res_split:
            continue
        a, b = r["full"], res_split[("base", s)]["full"]
        ca, cb = a["probs"].argmax(1), b["probs"].argmax(1)
        row = dict(split=ds.name, config=cfg, seed=s, n_clean=int((~nat).sum()), n_native_missing=int(nat.sum()),
                   clean_max_abs_dprob=float(np.abs(a["probs"] - b["probs"]).max(1)[~nat].max()) if (~nat).any() else None,
                   clean_max_abs_dy=float(np.abs(a["yhat"] - b["yhat"])[~nat].max()) if (~nat).any() else None)
        if nat.any():
            row.update(native_gate_trigger_rate=float((a["gF"][nat] > 1e-6).mean()), native_gateF_mean=float(a["gF"][nat].mean()),
                       native_gateT_mean=float(a["gT"][nat].mean()), native_class_changed=int((ca != cb)[nat].sum()),
                       native_mean_abs_dprob=float(np.abs(a["probs"] - b["probs"]).max(1)[nat].mean()))
            if ds.has_label:
                row.update(native_mae_cfg=float(np.abs(a["yhat"] - ds.y)[nat].mean()), native_mae_base=float(np.abs(b["yhat"] - ds.y)[nat].mean()),
                           native_acc_cfg=float((ca == ds.c)[nat].mean()), native_acc_base=float((cb == ds.c)[nat].mean()))
                row["native_d_mae"] = row["native_mae_cfg"] - row["native_mae_base"]
                row["native_d_acc"] = row["native_acc_cfg"] - row["native_acc_base"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"v3_audit_native_missing_{ds.name}.csv", index=False)
    return df


V2_MG2_SELECTED_EPOCH = {17: 0, 29: 6, 43: 0, 71: 0, 97: 1}      # 来自 v2 全量运行的训练记录（history_mg2_s*.csv）


def per_seed_panel_table(sums, rules_df):
    """逐种子、逐面板：argmax 口径（未校准）与规则层口径（校准 + 可能的决策层）并列，并注明 mg2 选中的 epoch。"""
    keep = ["complete_f1", "complete_acc", "complete_mae", "PM_avg_f1", "PM_worst_f1", "PM_avg_mae", "LH_avg_f1",
            "SC_avg_f1", "MS_avg_f1"]
    parts = []
    for sp, df in sums.items():
        d = df[["config", "seed"] + [c for c in keep if c in df.columns]].copy()
        d.insert(0, "口径", "argmax·未校准"); d.insert(0, "split", sp)
        parts.append(d)
    if rules_df is not None and len(rules_df):
        for pre, sp in (("vc_rules_", "valid_cal"), ("test_rules_", "test")):
            cols = {f"{pre}{c}": c for c in ["complete_f1", "complete_acc", "complete_mae", "PM_avg_f1", "PM_worst_f1",
                                               "PM_avg_mae", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"]}
            have = [c for c in cols if c in rules_df.columns]
            if have:
                d = rules_df[["config", "seed"] + have].rename(columns=cols).copy()
                d.insert(0, "口径", "规则层（VS 拟合）"); d.insert(0, "split", sp)
                parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t["mg2_selected_epoch_v2"] = [V2_MG2_SELECTED_EPOCH.get(int(s)) if c == "mg2" else None for c, s in zip(t.config, t.seed)]
    t.to_csv(OUT / "tables" / "v3_per_seed_panel.csv", index=False)
    return t


# %% [v3q3] 25. v3 · 问题三：删除检验改用绝对差值（带区间），声学/视觉报告分布而不只是均值
def faith_abs_summary(faith, tag, reps=4000):
    if faith is None or not len(faith):
        return pd.DataFrame()
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, np.float64)
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    rows = []
    for (m, b), g in faith.groupby(["modality", "budget"]):
        d, lo, hi = boot(g.del_top - g.del_rand)
        k, klo, khi = boot(g.keep_rand - g.keep_top)
        at = np.abs(g.del_top.values)
        rows.append(dict(modality=m, budget=b, n=len(g), del_top=float(g.del_top.mean()), del_rand=float(g.del_rand.mean()),
                         del_low=float(g.del_low.mean()), diff_top_minus_rand=d, diff_lo=lo, diff_hi=hi,
                         keep_gain_top_vs_rand=k, keep_lo=klo, keep_hi=khi,
                         top_beats_rand_share=float((g.del_top > g.del_rand).mean()),
                         abs_del_top_median=float(np.median(at)), abs_del_top_p90=float(np.percentile(at, 90)),
                         share_abs_del_top_gt_0_01=float((at > 0.01).mean()), share_abs_del_top_gt_0_05=float((at > 0.05).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q3" / f"v3_faithfulness_abs_{tag}.csv", index=False)
    return df


# %% [v3ext] 26. v3 · 扩展扫描（多配置、多种子，另算集成曲线）
def ext_scan_v3(preds, data):
    from scipy.stats import t as tdist
    cfgs = [c for c in CFG["V3_EXT_CONFIGS"] if any(k[0] == c for k in preds)]
    ens_fams = {n: f for n, f in CFG["ENSEMBLES"].items() if all(any(k[0] == x for k in preds) for x in f)}
    keep_pred = set(x for f in ens_fams.values() for x in f)
    rows, store = [], {}
    for sp in CFG["EXT_SPLITS"]:
        if sp not in data:
            continue
        ds = data[sp]
        panel = [x for x in build_panel(ds, "PM", ratios=CFG["EXT_RATIOS"], allow_clear=True) if x[1] in ("T", "A", "V", "AV", "TAV")]
        for cfg in cfgs:
            for key in sorted(k for k in preds if k[0] == cfg):
                for b, (nm, cb, r, pos, ms) in enumerate(panel):
                    p = preds[key].predict(ds, np.arange(ds.n), ms)
                    m = metrics(ds.c, ds.y, p["probs"], p["yhat"])
                    rows.append(dict(split=sp, config=cfg, seed=key[1], combo=cb, ratio=r, pos=pos, f1=m["f1"], acc=m["acc"],
                                     mae=m["mae"], rec_neu=m["rec_neu"], gateF=float(p["gF"].mean()), gateT=float(p["gT"].mean())))
                    if cfg in keep_pred:
                        store[(sp, key, b)] = {k: v for k, v in p.items() if k in ("probs", "yhat", "dist", "gT", "gF")}
            log(f"  扩展扫描 {sp} {cfg} 完成")
        for name, fams in ens_fams.items():
            for b, (nm, cb, r, pos, ms) in enumerate(panel):
                groups = [[store[(sp, k, b)] for k in sorted(preds) if k[0] == f] for f in fams]
                p = _family_combine(groups)
                m = metrics(ds.c, ds.y, p["probs"], p["yhat"])
                rows.append(dict(split=sp, config=name, seed=0, combo=cb, ratio=r, pos=pos, f1=m["f1"], acc=m["acc"], mae=m["mae"],
                                 rec_neu=m["rec_neu"], gateF=float(p["gF"].mean()), gateT=float(p["gT"].mean())))
        store = {k: v for k, v in store.items() if k[0] != sp}
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "v3_ext_scan_raw.csv", index=False)
    g = df.groupby(["split", "config", "seed", "combo", "ratio"])[["f1", "acc", "mae", "rec_neu", "gateF", "gateT"]].mean().reset_index()
    diffs = []
    for (sp, cb, r), d in g.groupby(["split", "combo", "ratio"]):
        base = d[d.config == "base"].set_index("seed")
        for cfg in [c for c in cfgs if c != "base"]:
            x = d[d.config == cfg].set_index("seed")
            ss = sorted(set(base.index) & set(x.index))
            if len(ss) < 2:
                continue
            for met in ("f1", "acc", "mae"):
                v = (x.loc[ss, met] - base.loc[ss, met]).values
                half = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
                diffs.append(dict(split=sp, combo=cb, ratio=r, compare=f"{cfg} - base", metric=met, mean=v.mean(),
                                  lo=v.mean() - half, hi=v.mean() + half, n_positive=int((v > 0).sum()), n=len(v)))
        for e1, e0 in (("deb_ens", "base_ens"), ("mix_ens", "base_ens")):
            a = d[d.config == e1]; b0 = d[d.config == e0]
            if len(a) and len(b0):
                for met in ("f1", "acc", "mae"):
                    diffs.append(dict(split=sp, combo=cb, ratio=r, compare=f"{e1} - {e0}", metric=met,
                                      mean=float(a[met].iloc[0] - b0[met].iloc[0]), lo=np.nan, hi=np.nan, n_positive=None, n=1))
    dd = pd.DataFrame(diffs)
    dd.to_csv(OUT / "tables" / "v3_ext_scan_diffs.csv", index=False)
    return df, g, dd


# %% [v3tgt] 27. v3 · 目标达成表（F1 ≥ 0.70、Acc-3 ≥ 0.70、MAE < 0.60）
def targets_table(rules_df, res, deploy, rules_by_key):
    """各配置在测试集（及校准区 VC）完整输入上、加规则层后的三项指标；部署模型另给按视频重抽样的 95% 区间。"""
    T = CFG["TARGETS"]
    rows = []
    for cfg, g in rules_df.groupby("config"):
        for sp, pre in (("test", "test_rules_"), ("valid_cal", "vc_rules_")):
            if f"{pre}complete_f1" not in g:
                continue
            f1, acc, mae = (float(g[f"{pre}complete_{k}"].mean()) for k in ("f1", "acc", "mae"))
            rows.append(dict(split=sp, config=cfg, n_models=len(g), f1=f1, acc=acc, mae=mae,
                             f1_ok=f1 >= T["f1"], acc_ok=acc >= T["acc"], mae_ok=mae < T["mae"]))
    df = pd.DataFrame(rows)
    ci = {}
    for sp in ("test", "valid_cal"):
        if sp in res and deploy in res[sp]:
            ds = DSETS[sp]; f = res[sp][deploy]["full"]
            c, y, _ = rules_by_key[deploy].apply(f["probs"], f["yhat"], full_meta(ds)[0])
            pt, lo, hi = video_boot(ds, lambda w: weighted_metrics(ds.c, ds.y, c, y, w), seed=13)
            ci[sp] = dict(f1=(pt[0], lo[0], hi[0]), acc=(pt[1], lo[1], hi[1]), mae=(pt[2], lo[2], hi[2]))
    df.to_csv(OUT / "tables" / "v3_targets.csv", index=False)
    save_json(ci, OUT / "tables" / "v3_targets_deploy_ci.json")
    return df, ci


def sums_from_res(res_split, ds):
    """由（切片后的）评估结果重建逐模型汇总行（与 evaluate_all 的行格式相同）。"""
    rows = []
    for (cfg, s), r in res_split.items():
        if "complete" not in r:
            continue
        row = dict(split=ds.name, config=cfg, seed=s)
        row.update({f"complete_{k}": v for k, v in r["complete"].items()})
        for kind in PANEL_KINDS:
            if kind + "_sum" in r:
                row.update({f"{kind}_{k}": v for k, v in r[kind + "_sum"].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"summary_runs_{ds.name}.csv", index=False)
    return df


def per_class_table(res_split, ds, keys):
    rows = []
    for k in keys:
        if k in res_split:
            m = res_split[k]["complete"]
            rows.append(dict(split=ds.name, model=f"{k[0]}" + ("" if k[1] == 0 else f"_s{k[1]}"), f1=m["f1"], acc=m["acc"],
                             mae=m["mae"], f1_neg=m["f1_neg"], f1_neu=m["f1_neu"], f1_pos=m["f1_pos"], rec_neg=m["rec_neg"],
                             rec_neu=m["rec_neu"], rec_pos=m["rec_pos"], prec_neu=m["prec_neu"], corr=m["corr"],
                             acc2=m["acc2"], acc7=m["acc7"], ece=m["ece"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"v3_per_class_{ds.name}.csv", index=False)
    return df

# %% [figures] 14. 作图（英文标签，避免 Kaggle 上缺中文字体）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
CFG_COLORS = dict(base="#7f7f7f", joint="#1f77b4", mg="#d62728", mg_nogate="#ff7f0e", mg_nokd="#2ca02c", mg_scatter="#9467bd")
CFG_ORDER = ["base", "joint", "mg_nogate", "mg_nokd", "mg_scatter", "mg"]
FIG = OUT / "figures"


def safe(fn):
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            import traceback
            log(f"  [作图跳过] {fn.__name__}: {e!r}")
            traceback.print_exc()
            plt.close("all")
    wrap.__name__ = fn.__name__
    return wrap


def _esc(x):
    return str(x).replace("$", r"\$")


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png"); plt.close(fig)


@safe
def fig_data_structure(data):
    tr = data["train"]
    order = np.argsort(tr.sep)[:: max(1, tr.n // 400)]
    panels = [("train: text observed", tr.oT[order]), ("train: audio observed", tr.oA[order]),
              ("train: vision observed", tr.oV[order])]
    if "att3" in data:
        a3 = data["att3"]
        panels += [("att3: audio observed", a3.oA), ("att3: vision observed", a3.oV)]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 3.2))
    for ax, (t, M) in zip(np.atleast_1d(axs), panels):
        ax.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_title(t); ax.set_xlabel("position"); ax.set_ylabel("sample")
    fig.suptitle("Observed (dark) vs zero (light) positions")
    _save(fig, "fig01_data_structure")


@safe
def fig_training_curves(dep):
    fs = sorted(glob.glob(str(OUT / "runs" / f"history_*_s{dep}.csv")))
    if not fs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for f in fs:
        nm = os.path.basename(f)[8:].rsplit("_s", 1)[0]
        h = pd.read_csv(f)
        axs[0].plot(h.epoch, h.f1, "-o", ms=3, label=nm, color=CFG_COLORS.get(nm))
        if "pm_f1" in h:
            axs[1].plot(h.epoch, h.pm_f1, "-o", ms=3, label=nm, color=CFG_COLORS.get(nm))
    axs[0].set_title("Valid complete-input macro-F1"); axs[1].set_title("Valid PM-panel mean macro-F1")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    _save(fig, "fig02_training_curves")


@safe
def fig_config_compare(sums):
    mets = ["complete_f1", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"]
    fig, axs = plt.subplots(1, len(sums), figsize=(5.2 * len(sums), 3.2), squeeze=False)
    for ax, (split, df) in zip(axs[0], sums.items()):
        cfgs = [c for c in CFG_ORDER if c in set(df.config)]
        w = 0.8 / len(cfgs)
        for j, c in enumerate(cfgs):
            d = df[df.config == c]
            mu = [d[m].mean() for m in mets]; sd = [d[m].std(ddof=1) if len(d) > 1 else 0 for m in mets]
            ax.bar(np.arange(len(mets)) + j * w, mu, w, yerr=sd, capsize=2, label=f"{c} (n={len(d)})", color=CFG_COLORS[c])
        ax.set_xticks(np.arange(len(mets)) + 0.4 - w / 2); ax.set_xticklabels(mets, rotation=20)
        lo = min(df[m].min() for m in mets); ax.set_ylim(max(0, lo - 0.05), None)
        ax.set_title(f"{split}: macro-F1 by condition (mean ± sd over seeds)"); ax.legend(fontsize=6, ncol=2)
    _save(fig, "fig03_config_compare")


@safe
def fig_pm_heatmap(res, dep, split):
    if ("mg", dep) not in res or ("base", dep) not in res:
        return
    def piv(k):
        d = res[k]["PM_df"].copy(); d["col"] = d.ratio.astype(str) + "/" + d.pos
        return d.pivot(index="combo", columns="col", values="f1").loc[COMBOS]
    B, M = piv(("base", dep)), piv(("mg", dep))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (t, D, cm) in zip(axs, [("base", B, "viridis"), ("MG-LoRA", M, "viridis"), ("MG-LoRA - base", M - B, "RdBu")]):
        v = np.nanmax(np.abs(D.values)) if cm == "RdBu" else None
        im = ax.imshow(D.values, aspect="auto", cmap=cm, vmin=-v if v else None, vmax=v)
        ax.set_yticks(range(len(D.index))); ax.set_yticklabels(D.index)
        ax.set_xticks(range(len(D.columns))); ax.set_xticklabels(D.columns, rotation=60, fontsize=6)
        ax.set_title(f"{split} PM macro-F1: {t}"); fig.colorbar(im, ax=ax, fraction=0.04)
        for (r_, c_), val in np.ndenumerate(D.values):
            ax.text(c_, r_, f"{val:.2f}" if cm != "RdBu" else f"{val:+.2f}", ha="center", va="center", fontsize=5,
                    color="w" if cm != "RdBu" else "k")
    _save(fig, f"fig04_pm_heatmap_{split}")


@safe
def fig_ext_scan(ext):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for nm, df in ext.items():
        g = df.groupby(["combo", "ratio"])[["f1", "mae"]].mean().reset_index()
        for cb, st in zip(["T", "A", "V", "AV", "TAV"], ["-o", "-s", "-^", "-v", "-d"]):
            d = g[g.combo == cb]
            ls = st if nm == "mg" else st.replace("-", "--")
            axs[0].plot(d.ratio, d.f1, ls, ms=3, label=f"{nm}:{cb}", color=CFG_COLORS.get(nm), alpha=0.5 + 0.1 * len(cb))
            axs[1].plot(d.ratio, d.mae, ls, ms=3, label=f"{nm}:{cb}", color=CFG_COLORS.get(nm), alpha=0.5 + 0.1 * len(cb))
    axs[0].set_ylabel("macro-F1"); axs[1].set_ylabel("MAE")
    for ax in axs:
        ax.set_xlabel("missing ratio within modality (1.0 = whole modality)"); ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Extended scan (valid): all-or-nothing behaviour of text")
    _save(fig, "fig05_extended_scan")


@safe
def fig_gate(gdf):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    d = gdf[gdf.panel == "PM"]
    mm = d[["miss_T", "miss_A", "miss_V"]].max(1)
    b = pd.cut(mm, [-0.01, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    g = d.groupby(b).gateF.agg(["mean", "std", "count"])
    axs[0].errorbar(range(len(g)), g["mean"], g["std"], fmt="-o", ms=3)
    axs[0].set_xticks(range(len(g))); axs[0].set_xticklabels([str(x) for x in g.index], rotation=45, fontsize=6)
    axs[0].set_title("fusion gate vs max missing fraction"); axs[0].set_ylabel("gate g_F")
    bt = pd.cut(d.miss_T, [-0.01, 0, 0.1, 0.2, 0.3, 0.5, 1.0])
    g2 = d.groupby(bt).gateT.agg(["mean", "std"])
    axs[1].errorbar(range(len(g2)), g2["mean"], g2["std"], fmt="-o", ms=3, color="C3")
    axs[1].set_xticks(range(len(g2))); axs[1].set_xticklabels([str(x) for x in g2.index], rotation=45, fontsize=6)
    axs[1].set_title("text gate vs text missing fraction")
    gc = d.groupby("combo").gateF.mean().loc[COMBOS]
    axs[2].bar(gc.index, gc.values, color="C1"); axs[2].set_title("mean fusion gate by masked modalities")
    _save(fig, "fig06_gate")


@safe
def fig_error_attr(tab, tag):
    facs = [f for f in ["true_class", "abs_y_bin", "len_bin", "conf_bin", "dom_group", "native_missing"] if f in set(tab.factor)]
    fig, axs = plt.subplots(1, len(facs), figsize=(2.3 * len(facs), 2.8))
    for ax, f in zip(np.atleast_1d(axs), facs):
        d = tab[tab.factor == f]
        ax.bar(d.level.astype(str), d.err_rate, color="C0")
        for x, (e, n) in enumerate(zip(d.err_rate, d.n)):
            ax.text(x, e, f"n={n}", ha="center", va="bottom", fontsize=6)
        ax.set_title(f); ax.tick_params(axis="x", rotation=45)
    np.atleast_1d(axs)[0].set_ylabel("error rate (3-class)")
    _save(fig, f"fig07_error_attribution_{tag}")


@safe
def fig_confusions(cms, name):
    fig, axs = plt.subplots(1, len(cms), figsize=(2.9 * len(cms), 2.6))
    for ax, (t, C) in zip(np.atleast_1d(axs), cms.items()):
        Cn = C / np.maximum(1, C.sum(1, keepdims=True))
        ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(C):
            ax.text(j, i, f"{int(v)}\n{Cn[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(3)); ax.set_xticklabels(["neg", "neu", "pos"]); ax.set_yticks(range(3)); ax.set_yticklabels(["neg", "neu", "pos"])
        ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title(t)
    _save(fig, name)


@safe
def fig_att3(det):
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))
    x = np.arange(len(det))
    col = det.polarity.map({"Negative": "C0", "Neutral": "C7", "Positive": "C3"})
    axs[0, 0].bar(x, det.intensity, color=col); axs[0, 0].axhline(0, color="k", lw=0.5)
    axs[0, 0].set_title("att3: predicted intensity (color = polarity)"); axs[0, 0].set_xlabel("sample")
    axs[0, 1].bar(x, det.confidence, color=col); axs[0, 1].axhline(CFG["REVIEW_PMAX"], ls="--", color="k", lw=0.6)
    axs[0, 1].set_title("att3: confidence (max class prob)")
    w = 0.28
    for j, m in enumerate(MODS):
        axs[1, 0].bar(x + (j - 1) * w, det[f"miss_frac_{m}"], w, label=m)
    axs[1, 0].set_title("att3: missing fraction per modality"); axs[1, 0].legend(fontsize=7)
    axs[1, 1].scatter(det[["miss_frac_T", "miss_frac_A", "miss_frac_V"]].max(1), det.gate_fusion, s=12, c=col)
    axs[1, 1].set_xlabel("max missing fraction"); axs[1, 1].set_ylabel("fusion gate"); axs[1, 1].set_title("att3: gate opens with missingness")
    _save(fig, "fig08_attachment3")


@safe
def fig_q3_modality(shap_va, df4):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].boxplot([shap_va[f"share_cls_{m}"].dropna() for m in MODS], labels=["text", "audio", "vision"], showfliers=False)
    axs[0].set_title("valid: Shapley share (classification head)")
    if df4 is not None and len(df4):
        x = np.arange(len(df4)); bottom = np.zeros(len(df4))
        for m, c in zip(MODS, ["C3", "C0", "C2"]):
            axs[1].bar(x, df4[f"contrib_{m}"], bottom=bottom, color=c, label=m); bottom += df4[f"contrib_{m}"].fillna(0).values
        axs[1].set_xticks(x); axs[1].set_xticklabels([_esc(s[-10:]) for s in df4.sample_id], rotation=90, fontsize=5)
        axs[1].set_title("att4: modality contribution per sample"); axs[1].legend(fontsize=7)
    _save(fig, "fig09_q3_modality_share")


@safe
def fig_q3_local(card, tag):
    fig, axs = plt.subplots(3, 1, figsize=(9, 5.2))
    for ax, m, c in zip(axs, MODS, ["C3", "C0", "C2"]):
        li = card["local_importance"][m]
        if not li["positions"]:
            ax.set_title(f"{m}: no observed positions"); continue
        ax.bar(li["positions"], li["score"], color=c)
        ev = card["evidence"][m]
        for e in ev:
            ax.axvspan(min(e["positions"]) - 0.5, max(e["positions"]) + 0.5, color="gold", alpha=0.3)
        if m == "T":
            ax.set_xticks(li["positions"]); ax.set_xticklabels([_esc(w) for w in li["words"]], rotation=70, fontsize=6)
        ax.set_ylabel(m)
    axs[0].set_title(f"{_esc(card['id'])}  pred={card['pred_class']} p={card['p_pred']:.2f}  "
                     f"share T/A/V = {card['share_cls']['T']:.2f}/{card['share_cls']['A']:.2f}/{card['share_cls']['V']:.2f}")
    axs[-1].set_xlabel("aligned position (word index)")
    _save(fig, f"fig10_local_{tag}")


@safe
def fig_q3_faith(fsum, tag):
    mods = [m for m in MODS if m in set(fsum.modality)]
    fig, axs = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 2.8), squeeze=False)
    for ax, m in zip(axs[0], mods):
        d = fsum[fsum.modality == m]
        for k, st in (("del_top", "-o"), ("del_rand", "--s"), ("del_low", ":^")):
            ax.plot(d.budget, d[k], st, ms=3, label=k)
        ax.set_title(f"{m}: drop of p(c*) after deletion"); ax.set_xlabel("deleted share"); ax.legend(fontsize=6)
    _save(fig, f"fig11_faithfulness_{tag}")


@safe
def fig_seed_diffs(sd_all):
    mets = ["complete_f1", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"]
    d = sd_all[(sd_all["compare"] == "mg - base") & (sd_all["metric"].isin(mets))]
    if not len(d):
        return
    splits = list(d.split.unique())
    fig, axs = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 2.8), squeeze=False)
    for ax, sp in zip(axs[0], splits):
        dd = d[d.split == sp].set_index("metric").loc[[m for m in mets if m in set(d[d.split == sp].metric)]]
        for x, (m, r) in enumerate(dd.iterrows()):
            ps = r.per_seed if isinstance(r.per_seed, list) else json.loads(str(r.per_seed))
            ax.scatter([x] * len(ps), ps, s=12, color="C3", alpha=0.7)
            ax.errorbar(x, r["mean"], [[r["mean"] - r.lo], [r.hi - r["mean"]]] if np.isfinite(r.lo) else None, fmt="k_", capsize=4)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(dd))); ax.set_xticklabels(dd.index, rotation=20)
        ax.set_title(f"{sp}: MG-LoRA - base per seed (95% t-CI)")
    _save(fig, "fig12_seed_diffs")


# %% [report] 15. 汇总报告
def _md(df, floatfmt=".4f"):
    if df is None or not len(df):
        return "（无）\n"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```\n"


def _cfg_table(agg):
    cols = [("complete_f1", "完整F1"), ("complete_acc", "完整Acc"), ("complete_mae", "完整MAE"), ("complete_corr", "完整Corr"),
            ("complete_acc2", "Acc-2"), ("complete_acc7", "Acc-7"), ("PM_avg_f1", "PM均F1"), ("PM_worst_f1", "PM最差F1"),
            ("PM_avg_mae", "PM均MAE"), ("LH_avg_f1", "LH均F1"), ("SC_avg_f1", "SC均F1"), ("MS_avg_f1", "MS均F1"), ("complete_ece", "ECE")]
    rows = []
    for cfg_name in [c for c in CFG_ORDER if c in agg.index]:
        r = {"配置": cfg_name, "种子数": int(agg.loc[cfg_name, "complete_f1_count"])}
        for k, nm in cols:
            mu, sd = agg.loc[cfg_name, f"{k}_mean"], agg.loc[cfg_name, f"{k}_std"]
            r[nm] = f"{mu:.4f}" + (f"±{sd:.4f}" if np.isfinite(sd) else "")
        rows.append(r)
    return pd.DataFrame(rows)


def write_report(S):
    L = ["# GMCM 2026 E 题 · MG-LoRA 运行报告\n",
         f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M')}，总用时 {(time.time() - T0) / 60:.1f} 分钟；SMOKE={CFG['SMOKE']}",
         f"- 部署种子（事先固定）：{CFG['DEPLOY_SEED']}；主配置种子 {CFG['SEEDS_MAIN']}，消融种子 {CFG['SEEDS_ABL']}",
         f"- 环境：{S.get('env')}\n"]
    L += ["## 1 数据审计\n", _md(S["audit"].round(4).T.reset_index().rename(columns={"index": "项"}), ".3f"),
          f"参照区间偏移：`{json.dumps(S.get('offsets'), ensure_ascii=False, default=_json_default)}`\n"]
    for split in ("valid", "test"):
        if split in S.get("agg", {}):
            L += [f"## 2 各配置汇总（{split}，均值±标准差）\n", _md(_cfg_table(S["agg"][split]))]
    for split in ("valid", "test"):
        if split in S.get("seed_diffs", {}):
            d = S["seed_diffs"][split]
            d = d[d.metric.isin(["complete_f1", "complete_mae", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1"])]
            L += [f"## 3 种子配对差值（{split}，A−B，95% t 区间）\n", _md(d.drop(columns=["per_seed"]))]
    for split in ("valid", "test"):
        if split in S.get("boot", {}):
            L += [f"## 4 视频配对重抽样（{split}，部署种子）\n", _md(S["boot"][split])]
    if "identity" in S:
        L += ["## 5 完整输入等价性（结构保证的实测）\n", _md(S["identity"])]
    if "gate_corr" in S:
        L += ["## 6 门控分析\n", f"`{json.dumps(S['gate_corr'], ensure_ascii=False, default=_json_default)}`\n"]
    if "law" in S:
        law = S["law"]
        L += ["## 7 缺失规律（验证集，MG-LoRA，部署种子）\n", _md(law[law.factor.isin(["combo", "ratio", "pos"])])]
    if "ext" in S:
        for nm, df in S["ext"].items():
            g = df.groupby(["combo", "ratio"])[["f1", "mae"]].mean().reset_index()
            L += [f"### 扩展扫描：{nm}\n", _md(g.pivot(index="combo", columns="ratio", values="f1").reset_index())]
    if "rules" in S:
        L += ["## 8 规则层（验证集按视频分组 5 折，折外结果）\n", _md(pd.DataFrame(S["rules"]["cv"])),
              f"决定：`{json.dumps({k: v for k, v in S['rules'].items() if k != 'cv'}, ensure_ascii=False, default=_json_default)}`\n"]
    if "err_tab" in S:
        L += ["## 9 验证集错误归因（部署模型，完整输入）\n", _md(S["err_tab"]), _md(S.get("err_pat"))]
    if "att3" in S:
        L += ["## 10 附件3\n", f"`{json.dumps(S['att3'], ensure_ascii=False, default=_json_default)}`\n"]
        if S.get("coverage") is not None:
            L += ["覆盖性检查（事后，只读附件3缺失描述量）：\n", _md(S["coverage"].drop(columns=["att3_quantile_in_family"], errors="ignore"))]
    if "q3_valid" in S:
        L += ["## 11 问题三（验证集）\n", f"`{json.dumps(S['q3_valid'], ensure_ascii=False, default=_json_default)}`\n",
              _md(S.get("faith_valid"))]
    if "att4" in S:
        L += ["## 12 附件4\n", f"`{json.dumps(S['att4'], ensure_ascii=False, default=_json_default)}`\n"]
        if S.get("att4_df") is not None:
            L += [_md(S["att4_df"][["sample_id", "polarity", "intensity", "confidence", "main_modality", "contrib_T",
                                    "contrib_A", "contrib_V", "text_evidence", "timing_method"]])]
    if "test_rules" in S:
        L += ["## 13 测试集：规则层效果\n", _md(S["test_rules"])]
    if "eff" in S and S["eff"] is not None and len(S["eff"]):
        L += ["## 14 效率\n", _md(S["eff"]), f"参数量：`{json.dumps(S.get('params'), default=_json_default)}`\n"]
    L += ["## 附：文件\n", "\n".join(f"- {p.relative_to(OUT)}" for p in sorted(OUT.rglob("*"))
                                     if p.is_file() and "ckpt" not in p.parts and p.suffix in (".csv", ".json", ".png", ".md"))]
    txt = "\n".join(L)
    (OUT / "REPORT.md").write_text(txt, encoding="utf-8")
    log("报告已写入 REPORT.md")
    return txt


def zip_outputs():
    zp = Path(CFG["OUT_DIR"]).parent / (Path(CFG["OUT_DIR"]).name + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and "ckpt" not in p.parts and not p.name.startswith("_tmp"):
                z.write(p, p.relative_to(OUT.parent))
    log(f"已打包 {zp}（{zp.stat().st_size / 1e6:.1f} MB，不含检查点）")
    return zp

# %% [v2fig] 21. v2 · 作图
CFG_COLORS.update(mg2="#8c564b")
CFG_ORDER = ["base", "joint", "mg_nogate", "mg_nokd", "mg_scatter", "mg", "mg2"]


@safe
def fig_v2_ext_scan(g, split):
    d = g[g.split == split]
    if not len(d):
        return
    combos = [c for c in ("T", "TAV", "A", "V") if c in set(d.combo)]
    fig, axs = plt.subplots(2, len(combos), figsize=(3.2 * len(combos), 5.2), squeeze=False)
    for j, cb in enumerate(combos):
        for cfg in CFG["EXT_CONFIGS"]:
            x = d[(d.combo == cb) & (d.config == cfg)].groupby("ratio")[["f1", "mae"]].agg(["mean", "std"])
            if not len(x):
                continue
            for i, met in enumerate(("f1", "mae")):
                axs[i, j].errorbar(x.index, x[(met, "mean")], x[(met, "std")], fmt="-o", ms=3, capsize=2,
                                   label=cfg, color=CFG_COLORS.get(cfg))
        axs[0, j].set_title(f"{split}: masked {cb}"); axs[1, j].set_xlabel("missing ratio (1.0 = whole modality)")
    axs[0, 0].set_ylabel("macro-F1 (mean ± sd over seeds)"); axs[1, 0].set_ylabel("MAE")
    axs[0, 0].legend(fontsize=7)
    _save(fig, f"fig21_ext_scan_multiseed_{split}")


@safe
def fig_v2_gate(gdfs):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for nm, gdf in gdfs.items():
        d = gdf[gdf.panel.isin(["PM", "SC", "MS"])]
        mm = d[["miss_T", "miss_A", "miss_V"]].max(axis=1)
        b = pd.cut(mm, [-0.01, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
        x = d.groupby(b).gateF.mean()
        axs[0].plot(range(len(x)), x.values, "-o", ms=3, label=nm, color=CFG_COLORS.get(nm))
        axs[0].set_xticks(range(len(x))); axs[0].set_xticklabels([str(i) for i in x.index], rotation=45, fontsize=6)
        bt = pd.cut(d.miss_T, [-0.01, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
        y = d.groupby(bt).gateT.mean()
        axs[1].plot(range(len(y)), y.values, "-o", ms=3, label=nm, color=CFG_COLORS.get(nm))
        axs[1].set_xticks(range(len(y))); axs[1].set_xticklabels([str(i) for i in y.index], rotation=45, fontsize=6)
    axs[0].set_title("fusion gate vs max missing fraction"); axs[1].set_title("text gate vs text missing fraction")
    axs[0].legend(fontsize=7)
    _save(fig, "fig22_gate_v1_vs_v2")


@safe
def fig_v2_rules(df):
    cfgs = [c for c in CFG_ORDER if c in set(df.config)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (pre, title) in zip(axs, (("valid", "valid (out-of-fold)"), ("test", "test"), ("test", "test: neutral recall (PM)"))):
        if pre + "_raw_J" not in df.columns:
            continue
        x = np.arange(len(cfgs)); w = 0.38
        if "neutral" in title:
            a = [df[df.config == c][f"{pre}_raw_PM_rec_neu"].mean() for c in cfgs]
            b = [df[df.config == c][f"{pre}_rules_PM_rec_neu"].mean() for c in cfgs]
        else:
            a = [df[df.config == c][f"{pre}_raw_J"].mean() for c in cfgs]
            b = [df[df.config == c][f"{pre}_rules_J"].mean() for c in cfgs]
        ax.bar(x - w / 2, a, w, label="argmax", color="#bbbbbb"); ax.bar(x + w / 2, b, w, label="Bayes decision layer", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(cfgs); ax.set_title(title); ax.set_ylim(min(a + b) - 0.03, max(a + b) + 0.02)
    axs[0].set_ylabel("J = ½ complete F1 + ½ mean(PM,SC,MS) F1"); axs[0].legend(fontsize=7)
    _save(fig, "fig23_decision_layer")


@safe
def fig_v2_conformal(cdf):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    for a, d in cdf.groupby("alpha"):
        axs[0].plot(d.bin, d.coverage_int, "-o", ms=3, label=f"intensity, target {1 - a:.0%}")
        axs[0].plot(d.bin, d.coverage_set, "--s", ms=3, label=f"polarity set, target {1 - a:.0%}")
        axs[0].axhline(1 - a, color="k", lw=0.5, ls=":")
        axs[1].plot(d.bin, d.width, "-o", ms=3, label=f"alpha={a}")
        axs[2].plot(d.bin, d.avg_set_size, "-o", ms=3, label=f"alpha={a}")
    axs[0].set_title("test coverage by missing severity"); axs[1].set_title("interval width"); axs[2].set_title("avg set size")
    for ax in axs:
        ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=6)
    _save(fig, "fig24_conformal")


@safe
def fig_v2_prop3(pdf, rho):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    for m in MODS:
        axs[0].plot(rho.index, rho[m], "-o", ms=3, label=m)
    axs[0].set_title("autocorrelation rho(k)"); axs[0].set_xlabel("lag"); axs[0].legend(fontsize=7)
    for ax, col, t in zip(axs[1:], ("pool_dev_empirical", "repr_dev", "flip_rate"),
                          ("pooled-mean deviation (sigma^2): emp. vs theory", "representation deviation (model)", "prediction flip rate")):
        for m, c in zip(MODS, ("C3", "C0", "C2")):
            d = pdf[pdf.modality == m]
            for mode, st in (("contig", "-o"), ("scatter", "--s")):
                dd = d[d["mode"] == mode]
                ax.plot(dd.ratio, dd[col], st, ms=3, color=c, label=f"{m}-{mode}")
                if col == "pool_dev_empirical":
                    ax.plot(dd.ratio, dd.pool_dev_theory, ":", color=c, lw=1)
        ax.set_title(t, fontsize=7); ax.set_xlabel("missing ratio")
    axs[1].legend(fontsize=5, ncol=2)
    _save(fig, "fig25_prop3_contig_vs_scatter")


@safe
def fig_v2_lowrank(ldf):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for sp, d in ldf.groupby("space"):
        ax.plot(d["rank"], d.energy, "-o", ms=3, label=f"{sp} (eff. rank {d.effective_rank.iloc[0]:.1f})")
        ax.plot(d["rank"], d.energy_random, ":", label=f"{sp} random baseline")
    ax.set_xscale("log", base=2); ax.set_xlabel("rank r"); ax.set_ylabel("cumulative energy")
    ax.set_title("missing-induced representation shift is low-rank"); ax.legend(fontsize=6)
    _save(fig, "fig26_lowrank")


@safe
def fig_v2_faith(fs_on, fs_off):
    fig, ax = plt.subplots(figsize=(5, 3))
    for nm, fs, st in (("gate active (v1 style)", fs_off, "--s"), ("gate frozen (v2)", fs_on, "-o")):
        if fs is None or not len(fs):
            continue
        g = fs.groupby("modality").top_beats_rand_share.mean()
        ax.plot(g.index, g.values, st, ms=4, label=nm)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_ylabel("share: deleting top > deleting random"); ax.set_title("local evidence faithfulness"); ax.legend(fontsize=7)
    _save(fig, "fig27_faithfulness_freeze")


@safe
def fig_v2_interactions(sva, s4):
    fig, ax = plt.subplots(figsize=(6, 3))
    pairs = ["I_cls_TA", "I_cls_TV", "I_cls_AV", "I_cls_TAV"]
    x = np.arange(len(pairs)); w = 0.38
    ax.bar(x - w / 2, [sva.loc[p, "mean_abs"] for p in pairs], w, label="valid |I|")
    if s4 is not None:
        ax.bar(x + w / 2, [s4.loc[p, "mean_abs"] for p in pairs], w, label="att4 |I|")
    ax.set_xticks(x); ax.set_xticklabels([p[6:] for p in pairs]); ax.set_title("Shapley interaction (classification head)")
    ax.legend(fontsize=7)
    _save(fig, "fig28_interactions")


# %% [v2rep] 22. v2 · 报告
def write_report_v2(S2):
    L = ["# MG-LoRA v2 运行报告\n",
         f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M')}，总用时 {(time.time() - T0) / 60:.1f} 分钟；SMOKE={CFG['SMOKE']}",
         f"- 环境：{S2.get('env')}",
         f"- 部署：`{json.dumps(S2.get('deploy'), ensure_ascii=False, default=_json_default)}`",
         f"- mg2 各种子选中的 epoch（0 表示“不补偿”胜出）：`{S2.get('mg2_selected_epoch')}`\n"]

    def tab(title, df, fmt=".4f"):
        L.extend([f"## {title}\n", _md(df, fmt)])

    if "agg_valid" in S2:
        tab("1 各配置（验证集，argmax，均值±标准差）", _cfg_table(S2["agg_valid"]))
    if "agg_test" in S2:
        tab("2 各配置（测试集，argmax）", _cfg_table(S2["agg_test"]))
    if "seed_valid" in S2:
        d = S2["seed_valid"]; d = d[d["compare"].str.startswith("mg2") & d["metric"].isin(
            ["complete_f1", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1", "PM_avg_mae"])]
        tab("3 mg2 种子配对差（验证集，argmax）", d.drop(columns=["per_seed", "split"], errors="ignore"))
    if "seed_test" in S2:
        d = S2["seed_test"]; d = d[d["compare"].str.startswith("mg2") & d["metric"].isin(
            ["complete_f1", "PM_avg_f1", "PM_worst_f1", "LH_avg_f1", "SC_avg_f1", "MS_avg_f1", "PM_avg_mae"])]
        tab("4 mg2 种子配对差（测试集，argmax）", d.drop(columns=["per_seed", "split"], errors="ignore"))
    if "rules_agg" in S2:
        a = S2["rules_agg"]
        keep = [c for c in a.columns if c.endswith("__mean") and any(k in c for k in (
            "_J", "complete_f1", "PM_avg_f1", "SC_avg_f1", "MS_avg_f1", "LH_avg_f1", "complete_mae", "PM_rec_neu"))]
        t = a[keep].copy(); t.columns = [c.replace("__mean", "") for c in keep]
        tab("5 贝叶斯决策层：各配置（valid 为折外，test 为验证集拟合后一次性应用）", t.reset_index())
    if "rules_diffs" in S2:
        d = S2["rules_diffs"]; d = d[d.metric.str.contains("test_rules") | d.metric.str.contains("valid_rules_J")]
        tab("6 决策层之后的种子配对差", d.drop(columns=["per_seed"]))
    if "identity" in S2:
        tab("7 完整输入等价性（mg / mg2）", S2["identity"])
    if "gate_coef" in S2 and len(S2["gate_coef"]):
        tab("8 v2 单调门控系数 a（越大表示该缺失量越能“打开”补偿）", S2["gate_coef"], ".3f")
    if "ext_diffs" in S2 and len(S2["ext_diffs"]):
        d = S2["ext_diffs"]; d = d[(d.metric == "f1") & d.combo.isin(["T", "TAV"])]
        tab("9 扩展扫描：相对 base 的 F1 差（种子配对 95% 区间）", d)
    if S2.get("conformal") is not None:
        tab("10 Mondrian 共形预测（测试集覆盖率）", S2["conformal"])
    if S2.get("prop3") is not None:
        tab("11 命题 3 数值检验（连续 vs 散点）", S2["prop3"])
        tab("11b 自相关 ρ(k)", S2["rho"].reset_index(), ".3f")
    if S2.get("lowrank") is not None:
        tab("12 低秩检验（缺失引起的表示偏移的累计能量）", S2["lowrank"])
    if S2.get("q3") is not None:
        L.append("## 13 问题三\n")
        for k, v in S2["q3"].items():
            L.append(f"- {k}：`{json.dumps(v, ensure_ascii=False, default=_json_default)}`")
        L.append("")
    if S2.get("interactions") is not None:
        tab("14 Shapley 交互（验证集）", S2["interactions"].reset_index())
    if S2.get("att3") is not None:
        L += ["## 15 附件3\n", f"`{json.dumps(S2['att3'], ensure_ascii=False, default=_json_default)}`\n"]
    if S2.get("att4") is not None:
        L += ["## 16 附件4\n", f"`{json.dumps(S2['att4'], ensure_ascii=False, default=_json_default)}`\n"]
    if S2.get("errors"):
        L += ["## 运行中跳过的步骤\n"] + [f"- {e}" for e in S2["errors"]] + [""]
    L += ["## 附：文件\n", "\n".join(f"- {p.relative_to(OUT)}" for p in sorted(OUT.rglob("*"))
                                     if p.is_file() and "ckpt" not in p.parts and p.suffix in (".csv", ".json", ".png", ".md"))]
    (OUT / "REPORT_v2.md").write_text("\n".join(L), encoding="utf-8")
    log("报告已写入 REPORT_v2.md")

# %% [v3fig] 28. v3 · 作图
CFG_COLORS.update(mg2="#8c564b", deb="#2ca02c", base_ens="#4d4d4d", deb_ens="#006d2c", mix_ens="#6a3d9a")
V3_ORDER = ["base", "joint", "mg", "mg2", "deb", "base_ens", "deb_ens", "mix_ens"]


@safe
def fig_v3_targets(tdf, ci):
    d = tdf[tdf.split == "test"].set_index("config")
    cfgs = [c for c in V3_ORDER if c in d.index]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    for ax, met, tgt, lab in zip(axs, ("f1", "acc", "mae"), (CFG["TARGETS"]["f1"], CFG["TARGETS"]["acc"], CFG["TARGETS"]["mae"]),
                                 ("macro-F1", "Acc-3", "MAE")):
        ax.bar(range(len(cfgs)), d.loc[cfgs, met], color=[CFG_COLORS.get(c, "C0") for c in cfgs])
        ax.axhline(tgt, ls="--", color="k", lw=0.8)
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs, rotation=30, fontsize=7)
        lo = d.loc[cfgs, met].min(); hi = d.loc[cfgs, met].max()
        ax.set_ylim(min(lo, tgt) - 0.03, max(hi, tgt) + 0.03); ax.set_title(f"test {lab} (after rules); dashed = target")
    _save(fig, "fig30_v3_targets")


@safe
def fig_v3_per_class(pc):
    d = pc.get("test")
    if d is None or not len(d):
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.8 / len(d)
    for j, (_, r) in enumerate(d.iterrows()):
        ax.bar(np.arange(3) + (j - (len(d) - 1) / 2) * w, [r.f1_neg, r.f1_neu, r.f1_pos], w, label=r.model)
    ax.set_xticks(range(3)); ax.set_xticklabels(["negative", "neutral", "positive"]); ax.set_ylabel("F1 (test, argmax)")
    ax.legend(fontsize=6, ncol=2); ax.set_title("Per-class F1 on test")
    _save(fig, "fig31_v3_per_class")


@safe
def fig_v3_ext(g):
    splits = [s for s in ("valid", "test") if s in set(g.split)]
    fig, axs = plt.subplots(1, len(splits), figsize=(5 * len(splits), 3.2), squeeze=False)
    for ax, sp in zip(axs[0], splits):
        d = g[(g.split == sp) & (g.combo == "T")]
        for cfg in [c for c in V3_ORDER if c in set(d.config)]:
            x = d[d.config == cfg].groupby("ratio").f1.agg(["mean", "std"]).reset_index()
            ax.errorbar(x.ratio, x["mean"], yerr=x["std"].fillna(0), fmt="-o", ms=3, capsize=2, label=cfg,
                        color=CFG_COLORS.get(cfg))
        ax.set_xlabel("text missing ratio (contiguous window)"); ax.set_ylabel("macro-F1"); ax.set_title(f"{sp}: text-only missing")
        ax.legend(fontsize=6)
    _save(fig, "fig32_v3_ext_scan_text")


@safe
def fig_v3_audit(strata):
    splits = [s for s in ("valid", "test") if strata.get(s) is not None]
    fig, axs = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 3.2), squeeze=False)
    for ax, sp in zip(axs[0], splits):
        d = strata[sp]; d = d[(d.ratio == 0.9) & (d.config != "base") & d.d_f1.notna()]
        labs = ["all"] + [b[2] for b in REM_BINS]
        cfgs = [c for c in ("mg", "joint", "mg2") if c in set(d.config)]
        w = 0.8 / max(1, len(cfgs))
        for j, cfg in enumerate(cfgs):
            x = d[d.config == cfg].set_index("stratum").reindex(labs)
            xs = np.arange(len(labs)) + (j - (len(cfgs) - 1) / 2) * w
            ax.bar(xs, x.d_f1, w, color=CFG_COLORS.get(cfg), label=cfg)
            ax.errorbar(xs, x.d_f1, yerr=[x.d_f1 - x.d_f1_video_lo, x.d_f1_video_hi - x.d_f1], fmt="none", ecolor="k", lw=0.7)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.set_xlabel("remaining content words (90% text removed)"); ax.set_ylabel("ΔF1 vs base (video 95% CI)")
        ax.set_title(f"{sp}: gain vs residual text"); ax.legend(fontsize=7)
    _save(fig, "fig33_v3_audit_residual_words")


@safe
def fig_v3_conformal(cdf):
    d = cdf[cdf.alpha == CFG["CONF_ALPHA"][0]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    x = np.arange(len(d))
    for ax, k, t in ((axs[0], "coverage_int", "intensity interval"), (axs[1], "coverage_set", "polarity set")):
        ax.errorbar(x, d[k], yerr=[d[k] - d[k + "_lo"], d[k + "_hi"] - d[k]], fmt="o", capsize=3)
        ax.axhline(1 - CFG["CONF_ALPHA"][0], ls="--", color="k", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(d.bin, rotation=15, fontsize=7); ax.set_title(f"test coverage: {t} (video CI)")
    _save(fig, "fig34_v3_conformal")


@safe
def fig_v3_faith(fa):
    tags = [t for t in ("fixed", "dynamic") if fa.get(t) is not None and len(fa[t])]
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, m in zip(axs, MODS):
        for j, t in enumerate(tags):
            d = fa[t][fa[t].modality == m]
            ax.errorbar(d.budget + (j - 0.5) * 0.01, d.diff_top_minus_rand, yerr=[d.diff_top_minus_rand - d.diff_lo, d.diff_hi - d.diff_top_minus_rand],
                        fmt="-o", ms=3, capsize=2, label={"fixed": "content effect (fixed gate)", "dynamic": "system total effect"}[t])
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{m}: Δp drop, top − random"); ax.set_xlabel("deleted share")
    axs[0].legend(fontsize=6)
    _save(fig, "fig35_v3_faithfulness_abs")


@safe
def fig_v3_deb_curves():
    fs = sorted(glob.glob(str(OUT / "runs" / "history_deb_s*.csv")))
    if not fs:
        return
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    for f in fs:
        h = pd.read_csv(f); s = os.path.basename(f)[len("history_deb_"):-4]
        for ax, k in zip(axs, ("f1", "acc", "mae")):
            ax.plot(h.epoch, h[k], "-o", ms=3, label=s)
            if "selected" in h:
                sel = h[h.selected == 1]
                ax.plot(sel.epoch, sel[k], "k*", ms=8)
    for ax, k in zip(axs, ("macro-F1", "Acc-3", "MAE")):
        ax.set_title(f"DeBERTa: VS {k} per epoch (* = selected)"); ax.set_xlabel("epoch")
    axs[0].legend(fontsize=6)
    _save(fig, "fig36_v3_deb_training")


def make_v3_figures():
    if S3.get("targets") is not None:
        fig_v3_targets(S3["targets"], S3.get("targets_ci"))
    if S3.get("per_class"):
        fig_v3_per_class(S3["per_class"])
    if "EXT_G" in globals():
        fig_v3_ext(EXT_G)
    if S3.get("audit_strata"):
        fig_v3_audit(S3["audit_strata"])
    if S3.get("conformal") is not None:
        fig_v3_conformal(S3["conformal"])
    if S3.get("faith_abs"):
        fig_v3_faith(S3["faith_abs"])
    fig_v3_deb_curves()


# %% [v3rep] 29. v3 · 报告
def _cfg_table_v3(agg):
    cols = [("complete_f1", "完整F1"), ("complete_acc", "完整Acc"), ("complete_mae", "完整MAE"), ("complete_corr", "Corr"),
            ("complete_f1_neu", "中性F1"), ("PM_avg_f1", "PM均F1"), ("PM_worst_f1", "PM最差F1"), ("LH_avg_f1", "LH均F1"),
            ("SC_avg_f1", "SC均F1"), ("MS_avg_f1", "MS均F1")]
    rows = []
    for c in [x for x in V3_ORDER if x in agg.index]:
        r = {"配置": c, "模型数": int(agg.loc[c, "complete_f1_count"])}
        for k, nm in cols:
            if f"{k}_mean" in agg.columns:
                mu, sd = agg.loc[c, f"{k}_mean"], agg.loc[c, f"{k}_std"]
                r[nm] = f"{mu:.4f}" + (f"±{sd:.4f}" if np.isfinite(sd) else "")
        rows.append(r)
    return pd.DataFrame(rows)


def write_report_v3(S):
    T = CFG["TARGETS"]
    L = ["# GMCM 2026 E 题 · v3 运行报告\n",
         f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M')}，总用时 {(time.time() - T0) / 60:.1f} 分钟；SMOKE={CFG['SMOKE']}",
         f"- 环境：{S.get('env')}；DeBERTa：{CFG['TEXT_MODEL']}（通用预训练，未用情感数据微调）",
         f"- 部署：**{DEPLOY_NAME}**（{S.get('deploy', {}).get('rule')}）\n"]
    L += ["## 0 事先锁定的协议（训练前写入 PROTOCOL_v3.json）\n", "```json\n" + json.dumps(S.get("protocol"), ensure_ascii=False, indent=1,
                                                                                    default=_json_default) + "\n```\n"]
    if S.get("targets") is not None:
        L += [f"## 1 目标达成（完整输入，加规则层；目标 F1 ≥ {T['f1']}、Acc-3 ≥ {T['acc']}、MAE < {T['mae']}）\n",
              "单模型为 5 个种子的均值；集成为一个模型。VC 对 DeBERTa 是未参与任何选择的留出数据。\n",
              _md(S["targets"])]
        ci = S.get("targets_ci") or {}
        for sp, d in ci.items():
            L.append(f"- 部署模型 {DEPLOY_NAME}（{sp}，按视频重抽样 95% 区间）：" +
                     "；".join(f"{k} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]" for k, v in d.items()))
        L.append("")
    for sp in ("valid", "valid_cal", "test"):
        if sp in S.get("agg", {}):
            L += [f"## 2 各配置汇总（{sp}，argmax 口径，未加规则层；均值±种子标准差）\n", _md(_cfg_table_v3(S["agg"][sp]))]
    if S.get("per_class"):
        for sp, d in S["per_class"].items():
            L += [f"### 逐类指标（{sp}）\n", _md(d)]
    for sp in ("test",):
        if sp in S.get("seed_diffs", {}):
            d = S["seed_diffs"][sp]
            d = d[d.metric.isin(["complete_f1", "complete_acc", "complete_mae", "PM_avg_f1", "PM_worst_f1", "SC_avg_f1", "MS_avg_f1"])]
            L += [f"## 3 种子配对差（{sp}，A−B，5 个种子的 t 区间：只反映训练随机性）\n", _md(d.drop(columns=["per_seed"]))]
    for sp, d in (S.get("boot") or {}).items():
        L += [f"## 4 视频配对重抽样（{sp}，{CFG['BOOT_REPS']} 次：反映样本不确定性）\n", _md(d)]
    if S.get("audit_strata"):
        L += ["## 5 v2 结论补验：90% 文本缺失档（只推理）\n",
              "按“删除后剩余的含字母数字整词数”分层；差值为与 base 的同种子配对差（d_*，种子 t 区间）及按视频重抽样区间（*_video_*）。"
              "joint 是关键对照：若它的收益与 mg 相当，收益应归因于“用缺失样本训练”，而非 MG 的结构。\n"]
        for sp, d in S["audit_strata"].items():
            if d is not None:
                cols = [c for c in ["ratio", "stratum", "config", "n_rows_per_seed", "n_seeds", "f1", "acc", "d_f1", "d_f1_seed_lo",
                                    "d_f1_seed_hi", "d_f1_video_lo", "d_f1_video_hi", "d_acc", "d_acc_video_lo", "d_acc_video_hi",
                                    "n_positive_f1"] if c in d.columns]
                L += [f"### {sp}\n", _md(d[d.ratio.isin([0.9, 1.0])][cols])]
    if S.get("native"):
        L += ["## 6 原生缺失子集（完整输入里本来就有缺失的样本）\n",
              "clean_*：无原生缺失子集上与同种子 base 的最大差值（mg/mg2 结构上应为 0；joint 是另训的模型，不适用）。"
              "native_*：门触发率、类别改变数、MAE/Acc 变化。\n"]
        for sp, d in S["native"].items():
            if d is not None:
                L += [f"### {sp}\n", _md(d)]
    if S.get("ext_diffs") is not None:
        d = S["ext_diffs"]; d = d[(d.combo == "T") & (d.metric == "f1")]
        L += ["## 7 扩展扫描：只删文本，与 base 的差（F1）\n", _md(d)]
    if S.get("rules_df") is not None:
        rd = S["rules_df"]
        cols = [c for c in ["config", "seed", "chosen", "use_dec", "vs_cv_raw_J", "vs_cv_rules_J", "vc_rules_complete_f1",
                            "vc_rules_complete_acc", "vc_rules_complete_mae", "test_rules_complete_f1", "test_rules_complete_acc",
                            "test_rules_complete_mae", "test_rules_PM_avg_f1", "test_rules_J"] if c in rd.columns]
        L += ["## 8 规则层（VS 拟合，VS 内按视频 5 折决定是否启用）与部署选择\n", _md(rd[cols]),
              f"部署选择：`{json.dumps(S.get('deploy'), ensure_ascii=False, default=_json_default)}`\n"]
    if S.get("conformal") is not None:
        L += ["## 9 共形预测（只在 VC 上校准；测试集覆盖率带按视频的区间）\n",
              "保证的前提是 VC 与测试样本可交换；样本按视频成簇。单元素集合比例（singleton）反映信息量。\n",
              _md(S["conformal"])]
    if S.get("q3"):
        L += ["## 10 问题三（验证集）\n",
              ("部署模型含门控：分别报告“固定门控的内容效应”和“动态门控的系统总效应”，两者回答不同的问题。\n" if S["q3"].get("gated_deploy")
               else "部署模型没有门控：删除输入不会触发任何补偿机制，“内容效应”与“系统总效应”是同一个量。\n"),
              "删除检验用绝对差值 Δ = 删高分位置的概率下降 − 删随机位置的概率下降（按样本 bootstrap 95% 区间）；"
              "声学/视觉同时报告 |删高分| 的中位数、90 分位与超过 0.01/0.05 的样本占比。\n"]
        for t, d in (S.get("faith_abs") or {}).items():
            L += [f"### 删除检验（{t}）\n", _md(d)]
        L += [f"`{json.dumps({k: v for k, v in S['q3']['valid'].items() if k != 'review_rule'}, ensure_ascii=False, default=_json_default)}`\n"]
        if S.get("interactions") is not None:
            L += ["交互项（负值只表示负向非加性，不能单凭符号认定信息冗余）：\n", _md(S["interactions"].reset_index())]
    if S.get("err_tab") is not None:
        L += ["## 11 验证集错误归因（部署模型，完整输入，含规则层）\n", _md(S["err_tab"]), _md(S.get("err_pat"))]
    if S.get("att3"):
        L += ["## 12 附件3\n", f"`{json.dumps(S['att3'], ensure_ascii=False, default=_json_default)}`\n"]
    if S.get("att4"):
        L += ["## 13 附件4\n",
              "时间戳状态：rejected（人工核验不通过，不给秒级定位）/ revised（人工修订）/ revised_pending（需修订但未填修订量）/"
              " flagged_unverified（对齐得分可疑，未核验）/ tool_unverified（工具时间戳，未人工核验）。\n",
              f"`{json.dumps(S['att4'], ensure_ascii=False, default=_json_default)}`\n"]
        if S.get("att4_df") is not None:
            cols = [c for c in ["sample_id", "polarity", "intensity", "confidence", "main_modality", "contrib_T", "contrib_A", "contrib_V",
                                "text_evidence", "text_time_s", "timing_status", "align_score", "align_null", "manual_note"]
                    if c in S["att4_df"].columns]
            L += [_md(S["att4_df"][cols])]
    if S.get("prop3"):
        L += ["## 14 命题 3：自相关估到滞后 8 与 20（连续与散点删除个数逐样本相同）\n"]
        for lag, d in S["prop3"].items():
            L += [f"### 滞后 {lag}\n", _md(d)]
    if S.get("deb_prep"):
        L += ["## 15 DeBERTa 词元映射与参数\n", _md(pd.DataFrame(S["deb_prep"]).T.reset_index().rename(columns={"index": "split"})),
              f"参数量：`{json.dumps(S.get('deb_params'), default=_json_default)}`\n"]
    if S.get("errors"):
        L += ["## 16 运行中被跳过的步骤\n", "\n".join(f"- {e}" for e in S["errors"]) + "\n"]
    L += ["## 附：文件\n", "\n".join(f"- {p.relative_to(OUT)}" for p in sorted(OUT.rglob("*"))
                                     if p.is_file() and "ckpt" not in p.parts and p.suffix in (".csv", ".json", ".png", ".md", ".gz"))]
    txt = "\n".join(L)
    (OUT / "REPORT_v3.md").write_text(txt, encoding="utf-8")
    log("报告已写入 REPORT_v3.md")
    return txt


def zip_ckpts():
    """新训的检查点（DeBERTa 与补训的 joint 等）另打一个包，方便以后上传复用；已有的旧检查点不重复打包。"""
    zp = Path(CFG["OUT_DIR"]).parent / (Path(CFG["OUT_DIR"]).name + "_ckpt.zip")
    new = set(S3.get("bert_trained_now", [])) | {f"deb_s{s}" for s in CFG["DEB_SEEDS"]}
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted((OUT / "ckpt").rglob("*")):
            if p.is_file() and (p.suffix == ".json" or p.stem in new):
                z.write(p, p.relative_to(OUT)); n += 1
    log(f"检查点已打包 {zp}（{n} 个文件，{zp.stat().st_size / 1e6:.1f} MB）")
    return zp

# %% [runA] A. 读数据 → 审计 → 评估面板（与 v1/v2 完全相同）→ 验证集按视频对半分 → 写入事先锁定的协议
data, STATS, OFF_INFO = load_all()
AUDIT = audit(data, OFF_INFO)
PRIOR = prior_of(data["train"])
_ynz = np.abs(data["train"].y[data["train"].y != 0])
EPS_Y = float(min(1 / 3, max(0.05, _ynz.min()))) if len(_ynz) else 0.1
PANELS = {}
for _k in ("valid", "test"):
    if _k in data:
        PANELS[_k] = dict(PM=build_panel(data[_k], "PM"), LH=build_panel(data[_k], "LH"), SC=build_sc_panel(data[_k]),
                          MS=build_ms_panel(data[_k]))
DEP = CFG["DEPLOY_SEED"]
VA, TE = data["valid"], data.get("test")
IDX_SEL, IDX_CAL = split_valid_by_video(VA)
PROTOCOL = dict(
    written_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    valid_split=dict(rule="按源视频号加盐 SHA-256 排序，前一半视频为校准区 VC，其余为选择区 VS（只看视频号）",
                     salt=CFG["VALID_SPLIT_SALT"], n_sel=int(len(IDX_SEL)), n_cal=int(len(IDX_CAL)),
                     videos_sel=int(len(set(np.array(VA.videos)[IDX_SEL]))), videos_cal=int(len(set(np.array(VA.videos)[IDX_CAL])))),
    deb_epoch_rule="DeBERTa 每个种子：VS 完整输入上 F1 + Acc − MAE 最大的 epoch",
    bert_models="base/mg/joint/mg2 沿用 v1/v2 的训练与选 epoch 方式（用整个验证集），所以 VC 对它们不是完全未触碰；对 v3 新训的 DeBERTa 是",
    rules_rule="每个模型在 VS 上按视频分组 5 折：决策层折外 J 增益 ≥ RULE_MIN_GAIN 才启用；校准/一致化同 v2 的判据",
    deploy_rule="VS 折外 J（½ 完整宏F1 + ½ PM/SC/MS 平均宏F1，含规则层）按配置平均最高者；单模型部署种子 17，集成记为种子 0",
    ensembles=CFG["ENSEMBLES"], ensemble_rule="类别概率平均；强度：成员都有分布则平均分布取中位数，否则平均点估计；多家族先族内平均再族间平均",
    conformal_rule="Mondrian 分桶（完整/≤20%/20–50%/>50%），只在 VC 上校准；每个样本每桶至多一行；测试覆盖率按视频重抽样给区间",
    test_policy="测试集只在全部选择完成后评估一次；若事后因测试结果修改任何设置，修改后的系统一律标为“探索性”",
    targets=CFG["TARGETS"], seeds=dict(bert=CFG["SEEDS_MAIN"], deb=CFG["DEB_SEEDS"], deploy=DEP),
    intervals="种子 t 区间 = 同种子配对差在种子间的 t(n−1) 区间（训练随机性）；视频区间 = 按源视频有放回重抽样（样本不确定性）。两者不混用",
)
save_json(PROTOCOL, OUT / "PROTOCOL_v3.json")
S3["protocol"] = PROTOCOL
log("面板：", {k: {kk: len(v) for kk, v in p.items()} for k, p in PANELS.items()})
log("协议已写入 PROTOCOL_v3.json：", PROTOCOL["valid_split"])

# %% [runB] B. BERT 系模型：读取 base / mg / joint（缺的 joint 种子补训）/ mg2（有检查点才读）
BERT_PATH = find_bert()
TOKENIZER = load_tokenizer(BERT_PATH)
PREDS, HISTS = run_training(data, plan_fn=v3_bert_plan)
try:
    import transformers
    S3["env"] = dict(torch=torch.__version__, transformers=transformers.__version__,
                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except Exception:
    S3["env"] = dict(python=sys.version.split()[0])
log("BERT 系模型：", sorted(PREDS))

# %% [runC] C. v2 结论补验（只推理）：90% 文本档逐样本、剩余词数分层、joint 对照
AUD_DF, AUD_STRATA = {}, {}
if CFG["RUN_AUDIT"]:
    for _sp in ("valid", "test"):
        if _sp in data:
            AUD_DF[_sp] = guard(f"补验逐样本（{_sp}）", severe_text_audit, PREDS, data[_sp], TOKENIZER)
            if AUD_DF[_sp] is not None:
                AUD_STRATA[_sp] = guard(f"补验分层（{_sp}）", audit_strata, AUD_DF[_sp], data[_sp])
    S3["audit_strata"] = AUD_STRATA
    for _sp, _tab in AUD_STRATA.items():
        if _tab is not None:
            log(f"补验（{_sp}，比例 0.9）：\n" + _tab[_tab.ratio == 0.9].round(4).to_string())

# %% [runD] D. DeBERTa-v3-large：词元映射 → 训练 5 个种子（或读取检查点）
DSETS = dict(valid=VA, test=TE)


def _deb_prepare_all():
    global DEB_PATH, DTOK
    DEB_PATH = find_deberta()
    DTOK = load_deb_tokenizer(DEB_PATH)
    return {k: prepare_deb(ds, TOKENIZER, DTOK) for k, ds in data.items()}


DEB_OK = False
if CFG["RUN_DEB"]:
    S3["deb_prep"] = guard("DeBERTa 分词与词元映射", _deb_prepare_all)
    DEB_OK = S3["deb_prep"] is not None
    log("DeBERTa 词元映射：", S3["deb_prep"])
VS = VA.subset(IDX_SEL, name="valid_sel"); VC = VA.subset(IDX_CAL, name="valid_cal")
PANELS["valid_sel"] = slice_panels(PANELS["valid"], IDX_SEL); PANELS["valid_cal"] = slice_panels(PANELS["valid"], IDX_CAL)
DSETS.update(valid_sel=VS, valid_cal=VC)
if DEB_OK:
    _r = guard("DeBERTa 训练", run_deb, data, VS)
    if _r is not None:
        DEB_PREDS, DEB_HISTS = _r
        PREDS.update(DEB_PREDS)
fig_training_curves(DEP)

# %% [runE] E. 全部单模型 × 验证/测试 × 四套面板；集成；切出 VS / VC；种子配对差与视频配对重抽样
EVAL_PREDS = {k: v for k, v in PREDS.items() if k[0] in ("base", "joint", "mg", "mg2", "deb")}
RES, SUMS = {}, {}
for _sp, _ds in (("valid", VA), ("test", TE)):
    if _ds is None:
        continue
    RES[_sp], SUMS[_sp], _ = evaluate_all(EVAL_PREDS, _ds)
    RES[_sp], SUMS[_sp] = add_ensembles(RES[_sp], SUMS[_sp], _ds)
RES["valid_sel"] = {k: slice_res(r, IDX_SEL, VA.n, VS) for k, r in RES["valid"].items()}
RES["valid_cal"] = {k: slice_res(r, IDX_CAL, VA.n, VC) for k, r in RES["valid"].items()}
SUMS["valid_cal"] = sums_from_res(RES["valid_cal"], VC)
AGGS = {sp: config_summary(SUMS, sp) for sp in SUMS}
S3["agg"] = AGGS
_bf = SUMS["valid"][SUMS["valid"].config == "base"].complete_f1
if not CFG["SMOKE"] and len(_bf) and _bf.mean() < 0.45:
    log("!!! 警告：base 在验证集上的完整 F1 只有 %.3f，读取的检查点很可能来自冒烟测试，结果不可用。" % _bf.mean())
S3["seed_diffs"] = {sp: seed_level_diffs(SUMS[sp], sp) for sp in ("valid", "test") if sp in SUMS}
ENS_PAIRS = [(("deb_ens", 0), ("base_ens", 0)), (("mix_ens", 0), ("base_ens", 0)), (("mix_ens", 0), ("deb_ens", 0)),
             (("deb_ens", 0), ("joint", DEP)), (("deb", DEP), ("base", DEP)), (("joint", DEP), ("base", DEP)),
             (("mg", DEP), ("base", DEP))]
S3["boot"] = {}
for _sp, _ds in (("test", TE), ("valid_cal", VC)):
    if _sp in RES:
        S3["boot"][_sp] = bootstrap_comparisons(RES[_sp], _ds, DEP, pairs=ENS_PAIRS, tag="_v3")
S3["native"] = {sp: guard(f"原生缺失补验（{sp}）", native_missing_audit, RES[sp], DSETS[sp]) for sp in ("valid", "test") if sp in RES}
_pc_keys = [("base", DEP), ("joint", DEP), ("deb", DEP), ("base_ens", 0), ("deb_ens", 0), ("mix_ens", 0)]
S3["per_class"] = {sp: per_class_table(RES[sp], DSETS[sp], _pc_keys) for sp in ("valid_cal", "test") if sp in RES}
for _sp, _tab in S3["per_class"].items():
    log(f"逐类指标（{_sp}）：\n" + _tab.round(4).to_string())

# %% [runF] F. 扩展扫描（缺失比例 0.1 → 1.0；多配置、多种子，集成另算）
if CFG["RUN_EXT_SCAN"]:
    _r = guard("扩展扫描", ext_scan_v3, EVAL_PREDS, data)
    if _r is not None:
        EXT_RAW, EXT_G, S3["ext_diffs"] = _r

# %% [runG] G. 规则层（VS 上拟合）→ 部署选择（VS 折外 J）→ 目标达成表
KEYS = list(RES["valid_sel"].keys())
RULES_BY_KEY, RULES_DF, S3["rules_agg"] = rules_all_models(RES, KEYS, EPS_Y)
S3["rules_df"] = RULES_DF
S3["rules_diffs"] = rules_seed_diffs(RULES_DF)
DEPLOY, S3["deploy"] = choose_deploy(RULES_DF)
ENS_PREDS = ensemble_predictors(EVAL_PREDS)
ALLP = {**EVAL_PREDS, **ENS_PREDS}
PRED = ALLP[DEPLOY]; RULES = RULES_BY_KEY[DEPLOY]
DEPLOY_NAME = DEPLOY[0] if DEPLOY[1] == 0 else f"{DEPLOY[0]}_s{DEPLOY[1]}"
S3["deploy"]["rules"] = {k: v for k, v in RULES.info.items() if k != "cv"}
S3["targets"], S3["targets_ci"] = targets_table(RULES_DF, RES, DEPLOY, RULES_BY_KEY)
log("目标达成（测试集，含规则层）：\n" + S3["targets"].round(4).to_string())
log("部署模型的按视频 95% 区间：", S3["targets_ci"])
S3["per_seed"] = per_seed_panel_table({sp: SUMS[sp] for sp in ("valid", "test") if sp in SUMS}, RULES_DF)

# %% [runH] H. Mondrian 共形预测：只在校准区 VC 上校准，测试集核验（带按视频的区间）
CONF = MondrianConformal().fit(RULES, RES["valid_cal"][DEPLOY], VC)
S3["conformal"] = CONF.evaluate(RULES, RES["test"][DEPLOY], TE) if TE is not None else None
if S3["conformal"] is not None:
    log("共形覆盖（测试集）：\n" + S3["conformal"].drop(columns=["set_composition"]).round(3).to_string())

# %% [runI] I. 问题三（部署模型）：固定门控的内容效应；有门控时另算动态门控的系统总效应
_members = [DEPLOY[0]] if DEPLOY[1] != 0 else CFG["ENSEMBLES"].get(DEPLOY[0], [])
_gated = any(GATE_MODE.get(c, "off") != "off" for c in _members)
SHAP_VA, FULL_VA = exact_shapley(PRED, VA)
_, _, CARDS_VA, FAITH_VA, STAB_VA, Q3V, _ = run_q3_valid(PRED, VA, TOKENIZER, freeze=True, shap_df=SHAP_VA, full=FULL_VA)
S3["q3"] = {"valid": Q3V, "gated_deploy": bool(_gated)}
S3["faith_abs"] = {"fixed": faith_abs_summary(FAITH_VA, "valid_fixed_gate")}
if _gated:
    _, _, _, FAITH_OFF, _, Q3V_OFF, _ = run_q3_valid(PRED, VA, TOKENIZER, freeze=False, tag="_dynamic",
                                                    shap_df=SHAP_VA, full=FULL_VA)
    S3["faith_abs"]["dynamic"] = faith_abs_summary(FAITH_OFF, "valid_dynamic_gate")
S3["interactions"] = guard("交互项（验证集）", interactions_summary, SHAP_VA, VA, "valid")
_c, _y, _ = RULES.apply(FULL_VA["probs"], FULL_VA["yhat"], full_meta(VA)[0])
_ea = guard("错误归因", error_attribution, VA, FULL_VA, SHAP_VA, "valid_v3", _c, _y)
if _ea is not None:
    S3["err_tab"], S3["err_pat"] = _ea[0], _ea[3]; fig_error_attr(S3["err_tab"], "valid_v3")
_ok = [c for c in CARDS_VA if CLASS_NAMES[VA.c[VA.ids.index(c["id"])]] == c["pred_class"]]
_bad = [c for c in CARDS_VA if CLASS_NAMES[VA.c[VA.ids.index(c["id"])]] != c["pred_class"]]
for _tag, _cs in (("valid_correct_v3", _ok), ("valid_error_v3", _bad)):
    if _cs:
        fig_q3_local(max(_cs, key=lambda c: c["p_pred"]), _tag)

# %% [runJ] J. 附件3（部署模型 + 规则层 + 共形区间）与附件4（解释 + 对齐置信度 + 人工核验）
if CFG["RUN_ATT3"] and "att3" in data:
    if DEPLOY[1] == 0:
        _mem = [(f"{k[0]}_s{k[1]}", EVAL_PREDS[k]) for k in sorted(EVAL_PREDS) if k[0] in _members]
        _mrules = {f"{k[0]}_s{k[1]}": RULES_BY_KEY.get(k) for k in sorted(EVAL_PREDS) if k[0] in _members}
    else:
        _mem = [(f"{k[0]}_s{k[1]}", EVAL_PREDS[k]) for k in sorted(EVAL_PREDS) if k[0] == DEPLOY[0]]
        _mrules = {f"{k[0]}_s{k[1]}": RULES_BY_KEY.get(k) for k in sorted(EVAL_PREDS) if k[0] == DEPLOY[0]}
    _bkey = ("base_ens", 0) if DEPLOY[1] == 0 and ("base_ens", 0) in ALLP else ("base", DEP)
    _r3 = guard("附件3", run_att3, PRED, RULES, data["att3"], DEPLOY_NAME, members=_mem, member_rules=_mrules,
                base_pred=ALLP.get(_bkey), base_rules=RULES_BY_KEY.get(_bkey), conf=CONF)
    if _r3 is not None:
        DET3, S3["att3"] = _r3; fig_att3(DET3)
DF4 = None
if CFG["RUN_Q3"] and "att4" in data:
    _r4 = guard("附件4", run_q3_att4, PRED, RULES, data["att4"], TOKENIZER, data, freeze=True)
    if _r4 is not None:
        DF4, CARDS4, SHAP4, IMPS4, S3["att4"] = _r4
        S3["att4_df"] = DF4
        guard("交互项（附件4）", interactions_summary, SHAP4, data["att4"], "att4")
        _pick = sorted(CARDS4, key=lambda c: c["p_pred"])
        for c in {id(x): x for x in (_pick[-1], _pick[0], _pick[len(_pick) // 2])}.values():
            fig_q3_local(c, "att4_v3_" + re.sub(r"[^0-9A-Za-z_-]", "_", c["id"]))
fig_q3_modality(SHAP_VA, DF4)
if TE is not None:
    _bk = ("base_ens", 0) if ("base_ens", 0) in RES["test"] else ("base", DEP)
    _fb = RES["test"][_bk]["full"]; _fd = RES["test"][DEPLOY]["full"]
    _ind_te = full_meta(TE)[0]
    fig_confusions({f"{_bk[0]} argmax": conf_mat(TE.c, _fb["probs"].argmax(1)),
                    f"{DEPLOY_NAME} argmax": conf_mat(TE.c, _fd["probs"].argmax(1)),
                    f"{DEPLOY_NAME} + rules": conf_mat(TE.c, RULES.apply(_fd["probs"], _fd["yhat"], _ind_te)[0])},
                   "fig29_confusion_test_v3")

# %% [runK] K. 命题 3：自相关估到滞后 8 与 20 各算一次理论值（基础模型，部署种子）
if CFG["RUN_PROP3"] and ("base", DEP) in PREDS:
    S3["prop3"] = {}
    for _j, _lag in enumerate(CFG["PROP3_MAXLAGS"]):
        _p3 = guard(f"命题3（滞后 {_lag}）", prop3_check, PREDS[("base", DEP)], VA, maxlag=_lag, model_level=(_j == 0))
        if _p3 is not None:
            S3["prop3"][_lag] = _p3[0]
            log(f"命题3（滞后 {_lag}）：\n" + _p3[0].round(4).to_string())

# %% [runL] L. 作图、报告、打包
guard("v3 作图", make_v3_figures)
SUMMARY = dict(protocol=S3.get("protocol"), deploy=S3.get("deploy"), targets=S3.get("targets"), targets_ci=S3.get("targets_ci"),
               att3=S3.get("att3"), att4=S3.get("att4"), q3=S3.get("q3"), deb_prep=S3.get("deb_prep"),
               deb_params=S3.get("deb_params"), errors=S3.get("errors"), minutes=(time.time() - T0) / 60)
guard("summary", save_json, {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in SUMMARY.items()},
      OUT / "summary_v3.json")
guard("报告", write_report_v3, S3)
zip_outputs()
if CFG["CKPT_ZIP"]:
    guard("检查点打包", zip_ckpts)
log("v3 全部完成。请下载 outputs_v3.zip 发回。")
