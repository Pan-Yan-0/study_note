"""A0：Q0 教师在 B2 生成行上的推理（Kaggle T4×2，fp16）。研究对齐 v4 / v4.1，冻结计划变更记录 11-12 条目。
v1.1（Codex 审阅 A0 v1 后）：所有输出共用一套概率校验；新任务、续跑和打包共用一套回执验收（绑定包清单、模型快照、推理环境、
当前 adapter 哈希）；全量入口两个 worker 都返回 0 才打包；复现回执补诊断字段。

  python3 teacher_a0.py setup                                   装固定版本的 transformers / peft / accelerate / datasets（上游 ensure_packages）
  python3 teacher_a0.py download --root R                       下载 Qwen3-4B-Instruct-2507 快照 cdbee75f（逐文件核对），写 R/model.json
  python3 teacher_a0.py smoke --root R --weights W [W ...]      fold 0 seed 42 的全部检查 + 前 128 条教师推理，打包 R/download/a0_smoke.zip
  python3 teacher_a0.py run --root R --weights W ... --folds 0,2,4
                                                                单个 worker；环境变量 A0_GPU 指定用哪张卡；逐折、逐种子
  python3 teacher_a0.py full --root R --weights W ...           两个 worker（GPU0：fold 0、2、4；GPU1：fold 1、3），都返回 0 才 finalize
  python3 teacher_a0.py finalize --root R --weights W ...       用同一套回执验收核对 15 份，打包 R/download/a0_results.zip

每个 (fold, seed)：
  1. 身份：adapter_model.safetensors、adapter_config.json 的哈希等于归档 COMPLETE.json；选中轮等于 trained.json；
     adapter_config 的 r / alpha / target_modules / dropout 等于上游常量。
  2. 载入：和上游 train() 相同的路径（get_peft_model → cast_mixed_precision_params → set_peft_model_state_dict），
     载入后逐张量和文件比对。
  3. 内层验证重评：输入和顺序同上游（dict.fromkeys 去重），一次 score() 调用（保留上游分批）；返回行数必须等于归档条数；
     argmax 和归档 val_probs/epoch{best}.json 的一致率 ≥ 98%。
  4. seed 42 另做逐折复现：按归档 cache.jsonl 的行序（即历史的"按长度降序、每 64 条一组"）重算全部输入；
     留出行 sym 标签和归档 pred_fold0_sym.csv 的一致率 ≥ 99%，四类概率最大绝对差 ≤ 0.10。
     另报平均差、差的分位数、最大差对应的输入和新旧概率、翻转样本的新旧 top-2 间距、低答案质量的行数、答案 token。
     本折 seed 42 没过门槛，这一折不出任何教师文件，进程以非零状态退出。
  5. 教师推理：B2 本折实际送训的生成行，两个方向去重（data/b2_inputs.json 的顺序），按两段长度之和降序（稳定排序）、
     每 64 条一组调用上游 score()。写 teacher/fold{k}/seed{s}.jsonl 和同名 .json 回执。
概率校验（复现、内层验证、教师推理和归档参考共用）：行数对得上；四类值都有限、非负、和为 1（容差 1e-4）；
  mass 有限且在 [0, 1 + 1e-6]；top_is_letter 是布尔值。JSON 写出一律 allow_nan=False。
回执验收（新任务写完、续跑跳过、打包时共用）：status、fold、seed、SMOKE / 正式、选中轮、两项 adapter 哈希（和当前文件比）、
  预期输入数、各检查的通过状态、输出文件哈希、代码和数据哈希、包清单摘要、模型快照摘要、推理环境（GPU 型号、dtype、torch、
  CUDA、固定依赖版本）都要对上。卡编号和模型缓存路径单独记录，不参与比较。
"""
import argparse, csv, hashlib, json, math, os, subprocess, sys, time, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "q4b"))
SEEDS = (42, 43, 44)
GATE = {"holdout_sym_agree": 0.99, "max_abs_diff": 0.10, "val_argmax_agree": 0.98}
CHUNK = 64
SMOKE_N = 128
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]
VERSION = "a0-teacher-v1.1"
PLAN = (("0", "0,2,4"), ("1", "1,3"))
ENV_KEYS = ("gpu", "torch", "cuda", "dtype", "model_revision", "transformers", "peft", "accelerate", "datasets")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def rj(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def wj(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(x, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8"); os.replace(t, p)


def die(msg, code=2):
    print("停止：" + msg, flush=True); raise SystemExit(code)


# ---------- 身份 ----------
def code_identity():
    names = ["teacher_a0.py", "q4b/upstream_qwen.py", "q4b/common.py", "q4b/worker.py", "q4b/expected_data.json"]
    return {n: sha(HERE / n) for n in names}


def data_identity(): return {"b2_inputs.json": sha(HERE / "data" / "b2_inputs.json")}


def bundle_identity():
    """包清单（含 Q0 归档、官方数据、B2 输入）先逐文件核对，再取清单本身的摘要。"""
    sums = HERE / "SHA256SUMS"
    if not sums.is_file(): die("包里缺 SHA256SUMS")
    for line in sums.read_text().splitlines():
        h, name = line.split("  ", 1)
        if not (HERE / name).is_file() or sha(HERE / name) != h: die("包文件和清单不一致：" + name)
    return sha(sums)


def model_identity(root):
    rec = rj(Path(root) / "model.json")
    return {"revision": rec["revision"], "files_digest": digest(rec["files"])}


def env_of(env): return {k: env.get(k) for k in ENV_KEYS}


def run_identity(root, env):
    return {"version": VERSION, "code": code_identity(), "data": data_identity(), "bundle": bundle_identity(),
            "model": model_identity(root), "env": env_of(env), "gate": GATE}


def archive(fold, seed): return HERE / "archive" / "Q0" / f"seed{seed}" / f"fold{fold}"
def argmax(p): return max(range(4), key=lambda k: p[k])
def symp(p, q): return [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]
def sort_inputs(pairs): return sorted(pairs, key=lambda x: -len(x[0]) - len(x[1]))
def chunks(seq, n=CHUNK): return [seq[i:i + n] for i in range(0, len(seq), n)]


# ---------- 概率校验 ----------
def valid_p(p):
    return (isinstance(p, (list, tuple)) and len(p) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in p)
            and abs(sum(p) - 1) <= 1e-4)


def check_results(res, n, what):
    if not isinstance(res, list) or len(res) != n: die(f"{what}：返回 {len(res) if isinstance(res, list) else '非列表'} 行，应为 {n}")
    for r in res:
        if not valid_p(r.get("p")): die(f"{what}：概率非法 {r.get('p')}")
        m = r.get("mass")
        if not isinstance(m, (int, float)) or isinstance(m, bool) or not math.isfinite(m) or not 0 <= m <= 1 + 1e-6: die(f"{what}：mass 非法 {m}")
        if not isinstance(r.get("top_is_letter"), bool): die(f"{what}：top_is_letter 不是布尔值")
    return res


def score_all(teacher, pairs, what):
    out = []
    for part in chunks(pairs):
        out.extend(check_results(teacher.score(part), len(part), what))
    return out


# ---------- 上游模块和模型 ----------
def upstream(root):
    import worker  # q4b/worker.py：load() 设置环境变量、加载上游模块、核对官方数据哈希
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    gpu = os.environ.get("A0_GPU", "0")
    q = worker.load(Path(root), HERE / "official", 0, False, 8)
    # 上游一导入就把 CUDA_VISIBLE_DEVICES 写成 "0"（单卡用）；CUDA 在第一次调用时才读它，所以这里改回本进程分到的那张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return q


def hf_verified(q, rev):
    """从 Hugging Face 按固定 commit 下载，逐文件核对（大文件 SHA-256，小文件 git blob 哈希）。"""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL, revision=rev, files_metadata=True)
    if info.sha != rev: die("Hugging Face 快照不同：" + str(info.sha))
    path = snapshot_download(MODEL, revision=rev)
    bad, checked = [], 0
    for sib in info.siblings:
        name = sib.rfilename
        if name in q.MODEL_SKIP_CHECK: continue
        f = Path(path) / name
        if not f.is_file(): bad.append(name + "：没下载到"); continue
        lfs = sib.lfs
        ok = (q.file_digest(str(f), "sha256") == (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256)) if lfs else \
             (q.file_digest(str(f), "sha1", b"blob %d\0" % f.stat().st_size) == sib.blob_id)
        checked += 1
        if not ok: bad.append(name + "：内容不同")
    if checked == 0 or bad: die("下载的文件和快照记录对不上：" + str(bad[:5]))
    return path, rev, "huggingface（按快照逐文件核对）"


def download(root):
    q = upstream(root)
    try:
        path, rev, src = hf_verified(q, REV)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"Hugging Face 没成功（{type(e).__name__}: {str(e)[:200]}），改用 ModelScope", flush=True)
        path, rev, src = q.fetch_from_modelscope(REV)
    if rev != REV: die("模型快照不同")
    p = Path(path)
    wj(Path(root) / "model.json", {"path": str(p), "revision": rev, "source": src,
                                   "files": {str(f.relative_to(p)): sha(f) for f in sorted(p.rglob("*")) if f.is_file()}})
    print("模型已下载并核对：", path, flush=True)


class Teacher:
    """上游 QwenSFT 外面包一层：载入一份 Q0 adapter（和上游 train() 的载入路径相同），用上游 score() 打分。"""

    def __init__(self, root):
        self.q = q = upstream(root)
        rec = rj(Path(root) / "model.json")
        if rec["revision"] != REV: die("model.json 的快照不对")
        q.fetch_model = lambda revision: (rec["path"], REV, rec["source"])
        self.b = q.QwenSFT(REV)
        self.env = self.b.info()
        self.answer_tokens = {"option_ids": self.b.option_ids, "target_ids": self.b.target_ids}

    def load(self, adapter_dir):
        import torch
        from peft import LoraConfig, TaskType, cast_mixed_precision_params, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        q, b = self.q, self.b
        cfg = rj(Path(adapter_dir) / "adapter_config.json")
        want = {"r": q.LORA_R, "lora_alpha": q.LORA_ALPHA, "lora_dropout": 0.0, "target_modules": sorted(q.LORA_TARGETS)}
        got = {"r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "lora_dropout": cfg["lora_dropout"], "target_modules": sorted(cfg["target_modules"])}
        if got != want: die(f"adapter_config 和上游常量不同：{got} / {want}")
        b.load_base()
        model = get_peft_model(b.model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=q.LORA_R, lora_alpha=q.LORA_ALPHA, lora_dropout=0.0,
                                                   bias="none", target_modules=q.LORA_TARGETS))
        cast_mixed_precision_params(model, dtype=b.dtype)
        saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
        expected = get_peft_model_state_dict(model)
        if set(saved) != set(expected) or any(tuple(saved[k].shape) != tuple(expected[k].shape) for k in saved):
            die("adapter 的键或形状和模型对不上")
        set_peft_model_state_dict(model, saved)
        now = get_peft_model_state_dict(model)
        differ = [k for k in saved if not torch.equal(now[k].detach().cpu().to(saved[k].dtype), saved[k])]
        if differ: die(f"载入后 {len(differ)} 个张量和文件不同")
        b.model = model
        return {"tensors": len(saved), "dtype": str(b.dtype)}

    def score(self, pairs): return self.b.score(pairs)
    def release(self): self.b.release()


# ---------- 单个任务 ----------
def adapter_dir(weights, fold, seed, best):
    rel = Path("formal") / "Q0" / f"seed{seed}" / f"fold{fold}" / "adapters" / f"epoch{best}"
    hits = [Path(w) / rel for w in weights if (Path(w) / rel / "adapter_model.safetensors").is_file()]
    hits += [p for w in weights for p in Path(w).glob("*/" + str(rel)) if (p / "adapter_model.safetensors").is_file()]
    if not hits: die(f"找不到 Q0 seed{seed} fold{fold} 第 {best} 轮的 adapter（{rel}）")
    return hits[0]


def identity(weights, fold, seed):
    a = archive(fold, seed)
    best = rj(a / "trained.json")["best_epoch"]
    files = rj(a / "COMPLETE.json")["files"]
    d = adapter_dir(weights, fold, seed, best)
    got = {n: sha(d / n) for n in ("adapter_model.safetensors", "adapter_config.json")}
    want = {n: files[f"adapters/epoch{best}/{n}"] for n in got}
    if got != want: die(f"fold{fold} seed{seed} 的 adapter 哈希和归档不同")
    return d, best, got


def val_rows(fold, seed, byid):
    return [byid[i] for i in rj(archive(fold, seed) / "actual_split.json")["val"]]


def val_inputs(fold, seed, byid):
    return list(dict.fromkeys((r["text1"], r["text2"]) for r in val_rows(fold, seed, byid)))


def recheck_val(teacher, fold, seed, best, byid):
    inputs = val_inputs(fold, seed, byid)
    old = rj(archive(fold, seed) / "val_probs" / f"epoch{best}.json")
    if len(old) != len(inputs) or not all(valid_p(p) for p in old): die("归档内层验证概率条数不对或非法")
    new = [x["p"] for x in check_results(teacher.score(inputs), len(inputs), f"fold{fold} seed{seed} 内层验证")]
    agree = sum(argmax(p) == argmax(o) for p, o in zip(new, old)) / len(old)
    diff = max(abs(a - b) for p, o in zip(new, old) for a, b in zip(p, o))
    return {"rows": len(old), "returned": len(new), "argmax_agree": agree, "max_abs_diff": diff, "pass": agree >= GATE["val_argmax_agree"]}


def top2_gap(p):
    s = sorted(p, reverse=True); return s[0] - s[1]


def quantiles(xs):
    s = sorted(xs); n = len(s)
    return {q: s[min(n - 1, int(q * n))] for q in (0.5, 0.9, 0.99)} | {"max": s[-1]}


def reproduce(teacher, fold, byid):
    a = archive(fold, 42)
    lines = [json.loads(l) for l in open(a / "cache.jsonl", encoding="utf-8") if l.strip()]
    if not all(valid_p(x["p"]) for x in lines): die("归档缓存的概率非法")
    pairs = [(x["a"], x["b"]) for x in lines]
    new = score_all(teacher, pairs, f"fold{fold} 复现")  # 按归档行序每 64 条一组，即历史分批
    per = [max(abs(p - o) for p, o in zip(r["p"], x["p"])) for x, r in zip(lines, new)]
    diffs = [abs(p - o) for x, r in zip(lines, new) for p, o in zip(r["p"], x["p"])]
    worst = max(range(len(per)), key=lambda i: per[i])
    flips = [i for i, (x, r) in enumerate(zip(lines, new)) if argmax(r["p"]) != argmax(x["p"])]
    prob = {(x["a"], x["b"]): r["p"] for x, r in zip(lines, new)}
    test = [byid[i] for i in rj(a / "actual_split.json")["test"]]
    arch = {r["instance_id"]: r["label"] for r in csv.DictReader(open(a / "pred_fold0_sym.csv", encoding="utf-8"))}
    lab = {r["instance_id"]: LABELS[argmax(symp(prob[(r["text1"], r["text2"])], prob[(r["text2"], r["text1"])]))] for r in test}
    if set(lab) != set(arch): die("留出行和归档预测对不上")
    agree = sum(lab[i] == arch[i] for i in lab) / len(lab)
    out = {"inputs": len(pairs), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs), "abs_diff_quantiles": quantiles(diffs),
           "argmax_flips": len(flips), "holdout_rows": len(lab), "holdout_sym_agree": agree, "holdout_sym_disagree": sum(lab[i] != arch[i] for i in lab),
           "worst": {"a": lines[worst]["a"], "b": lines[worst]["b"], "old_p": lines[worst]["p"], "new_p": new[worst]["p"]},
           "flipped": [{"a": lines[i]["a"], "b": lines[i]["b"], "old_top2_gap": top2_gap(lines[i]["p"]), "new_top2_gap": top2_gap(new[i]["p"]),
                        "old_p": lines[i]["p"], "new_p": new[i]["p"]} for i in flips[:30]],
           "low_mass_rows": sum(r["mass"] < 0.9 for r in new), "answer_tokens": getattr(teacher, "answer_tokens", None)}
    out["pass"] = agree >= GATE["holdout_sym_agree"] and out["max_abs_diff"] <= GATE["max_abs_diff"]
    return out


def official_rows():
    p = HERE / "official" / "final_data" / "train" / "dico_nli_train_track1_participant_labeled.csv"
    return {r["instance_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def expected_inputs(fold, smoke, b2):
    n = len(b2[str(fold)]); return min(n, SMOKE_N) if smoke else n


def accept(r, out, fold, seed, smoke, best, adapter_sha, n_inputs, n_val, rid):
    """新任务写完、续跑跳过、打包时共用的回执验收。返回问题列表（空 = 通过）。"""
    p = []
    want = {"status": "pass", "fold": fold, "seed": seed, "smoke": smoke, "best_epoch": best, "adapter_sha256": adapter_sha,
            "inputs": n_inputs, "run_identity": rid}
    for k, v in want.items():
        if r.get(k) != v: p.append(f"{k} 不一致")
    vr = r.get("val_recheck") or {}
    if not (vr.get("pass") is True and vr.get("rows") == n_val and vr.get("returned") == n_val and vr.get("argmax_agree", 0) >= GATE["val_argmax_agree"]):
        p.append("内层验证重评未通过或行数不对")
    if seed == 42:
        rp = r.get("reproduction") or {}
        n_cache = sum(1 for l in open(archive(fold, 42) / "cache.jsonl", encoding="utf-8") if l.strip())
        if not (rp.get("pass") is True and rp.get("inputs") == n_cache and rp.get("holdout_sym_agree", 0) >= GATE["holdout_sym_agree"]
                and rp.get("max_abs_diff", 1) <= GATE["max_abs_diff"]):
            p.append("逐折复现未通过或输入数不对")
    elif "reproduction" in r:
        p.append("非 seed 42 的回执不应有复现")
    if not out.exists() or sha(out) != r.get("output_sha256"): p.append("输出文件哈希不对")
    elif sum(1 for _ in open(out, encoding="utf-8")) != n_inputs: p.append("输出行数不对")
    return p


def task_paths(root, fold, seed, smoke):
    base = Path(root) / ("smoke" if smoke else "teacher") / f"fold{fold}"
    return base / f"seed{seed}.jsonl", base / f"seed{seed}.json"


def run_task(teacher, root, weights, fold, seed, byid, b2, smoke, rid):
    out, rec = task_paths(root, fold, seed, smoke)
    d, best, hashes = identity(weights, fold, seed)  # 续跑也先核对当前权重和归档
    n_inputs, n_val = expected_inputs(fold, smoke, b2), len(val_inputs(fold, seed, byid))
    if rec.exists():
        r = rj(rec)
        if r.get("status") != "pass": die(f"{rec} 记录为未通过；先看原因，不自动重跑")
        bad = accept(r, out, fold, seed, smoke, best, hashes, n_inputs, n_val, rid)
        if bad: die(f"{rec} 不能沿用：" + "；".join(bad) + "。不覆盖，请检查")
        print(f"跳过（已完成，回执验收通过）：fold{fold} seed{seed}", flush=True); return
    t0 = time.time()
    load = teacher.load(d)
    receipt = {"version": VERSION, "fold": fold, "seed": seed, "smoke": smoke, "best_epoch": best, "adapter_sha256": hashes, "load": load,
               "run_identity": rid, "device": {"A0_GPU": os.environ.get("A0_GPU", "0"), "model_path": rj(Path(root) / "model.json")["path"]}}
    receipt["val_recheck"] = recheck_val(teacher, fold, seed, best, byid)
    if seed == 42: receipt["reproduction"] = reproduce(teacher, fold, byid)
    if not (receipt["val_recheck"]["pass"] and receipt.get("reproduction", {"pass": True})["pass"]):
        receipt["status"] = "fail"; wj(rec, receipt)
        die(f"fold{fold} seed{seed} 没过门槛：复现 {json.dumps({k: v for k, v in receipt.get('reproduction', {}).items() if k not in ('flipped', 'worst')}, ensure_ascii=False)}"
            f"；内层验证 {receipt['val_recheck']}", 3)
    pairs = sort_inputs([tuple(x) for x in b2[str(fold)]])[:n_inputs]
    t1 = time.time()
    res = score_all(teacher, pairs, f"fold{fold} seed{seed} 教师推理")
    tmp = out.with_suffix(".jsonl.tmp"); out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for (a, b), r in zip(pairs, res):
            f.write(json.dumps({"a": a, "b": b, "p": r["p"], "mass": r["mass"], "top_is_letter": r["top_is_letter"]}, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, out)
    receipt.update(status="pass", inputs=len(pairs), output_sha256=sha(out), low_mass_rows=sum(r["mass"] < 0.9 for r in res),
                   seconds={"total": time.time() - t0, "teacher_inference": time.time() - t1}, mean_answer_mass=sum(r["mass"] for r in res) / len(res))
    bad = accept(receipt, out, fold, seed, smoke, best, hashes, n_inputs, n_val, rid)
    if bad: die("新写的回执没通过验收：" + "；".join(bad))
    wj(rec, receipt)
    print(f"完成：fold{fold} seed{seed}  内层验证一致 {receipt['val_recheck']['argmax_agree']:.3f}  "
          + (f"复现 sym 一致 {receipt['reproduction']['holdout_sym_agree']:.3f} 最大差 {receipt['reproduction']['max_abs_diff']:.3g}  " if seed == 42 else "")
          + f"教师 {len(pairs)} 条  用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)


def run(root, weights, folds, smoke, teacher_cls=Teacher):
    byid = official_rows()
    b2 = rj(HERE / "data" / "b2_inputs.json")["inputs"]
    teacher = teacher_cls(root)
    rid = run_identity(root, teacher.env)
    try:
        for fold in folds:
            for seed in (SEEDS[:1] if smoke else SEEDS):  # 每折先 seed 42（含复现门槛），过了才跑 43、44
                run_task(teacher, root, weights, fold, seed, byid, b2, smoke, rid)
            if smoke: break
    finally:
        teacher.release()


def pack(root, name, keep):
    root = Path(root); dest = root / "download"; dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest / name, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if p.is_file() and (rel.parts[0] in keep or str(rel) in ("A0_SUMMARY.json", "A0_FAILED.json", "model.json")):
                z.write(p, str(rel))
        for n in code_identity(): z.write(HERE / n, "source/" + n)
    print("已打包：", dest / name, flush=True)
    return dest / name


def finalize(root, weights):
    """打包前用同一套回执验收核对 15 份：当前 adapter 哈希、包清单、模型快照，以及所有任务的推理环境一致。"""
    root = Path(root); byid = official_rows(); b2 = rj(HERE / "data" / "b2_inputs.json")["inputs"]
    problems, rids = [], []
    for fold in range(5):
        for seed in SEEDS:
            out, rec = task_paths(root, fold, seed, False)
            if not rec.exists(): problems.append(f"缺 fold{fold} seed{seed}"); continue
            r = rj(rec); _, best, hashes = identity(weights, fold, seed)
            bad = accept(r, out, fold, seed, False, best, hashes, expected_inputs(fold, False, b2), len(val_inputs(fold, seed, byid)), r.get("run_identity"))
            problems += [f"fold{fold} seed{seed}：{b}" for b in bad]
            rids.append(r.get("run_identity"))
    if rids:
        if any(x != rids[0] for x in rids): problems.append("各任务的运行身份（代码、包、模型、推理环境）不一致")
        cur = {k: v for k, v in (rids[0] or {}).items() if k != "env"}
        now = {"version": VERSION, "code": code_identity(), "data": data_identity(), "bundle": bundle_identity(),
               "model": model_identity(root), "gate": GATE}
        if cur != now: problems.append("回执的代码、包或模型和当前不一致")
    summary = {"version": VERSION, "problems": problems, "run_identity": rids[0] if rids else None}
    wj(root / "A0_SUMMARY.json", summary)
    if problems: die("还不能打包：" + "；".join(problems[:10]))
    return pack(root, "a0_results.zip", ("teacher", "logs"))


def full(root, weights, spawn=subprocess.Popen, poll_seconds=120):
    """两个 worker 并行；都返回 0 才 finalize。有非零的写 A0_FAILED.json 并只打诊断包。"""
    root = Path(root); (root / "logs").mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-I", str(HERE / "teacher_a0.py")]
    procs = []
    for gpu, folds in PLAN:
        log = open(root / "logs" / f"worker{gpu}.log", "a")
        procs.append((gpu, spawn(py + ["run", "--root", str(root), "--weights", *weights, "--folds", folds], env=dict(os.environ, A0_GPU=gpu),
                                 stdout=log, stderr=subprocess.STDOUT, text=True), log))
    t0 = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(poll_seconds)
        print(f"{(time.time() - t0) / 60:.0f} 分钟：已完成 {len(list(root.glob('teacher/fold*/seed*.jsonl')))} / 15 个教师文件", flush=True)
    for _, _, log in procs: log.close()
    codes = {gpu: p.returncode for gpu, p, _ in procs}
    if any(c != 0 for c in codes.values()):
        wj(root / "A0_FAILED.json", {"version": VERSION, "worker_exit_codes": codes})
        pack(root, "a0_failed_diagnostics.zip", ("teacher", "logs"))
        die(f"有 worker 没正常结束：{codes}。不打结果包；请把 a0_failed_diagnostics.zip 发给 Claude", 4)
    return finalize(root, weights)


def smoke(root, weights):
    os.environ.setdefault("A0_GPU", "0")
    run(root, weights, [0], True)
    r = rj(Path(root) / "smoke" / "fold0" / "seed42.json")
    print(json.dumps({k: r[k] for k in ("status", "val_recheck", "load", "seconds")} |
                     {"reproduction": {k: v for k, v in r["reproduction"].items() if k not in ("flipped",)}}, ensure_ascii=False, indent=1))
    return pack(root, "a0_smoke.zip", ("smoke", "logs"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["setup", "download", "smoke", "run", "full", "finalize"])
    p.add_argument("--root"); p.add_argument("--weights", nargs="*", default=[]); p.add_argument("--folds", default="0,1,2,3,4")
    a = p.parse_args()
    if a.cmd == "setup":
        import importlib.util as u
        spec = u.spec_from_file_location("up", HERE / "q4b" / "upstream_qwen.py"); m = u.module_from_spec(spec); spec.loader.exec_module(m)
        m.ensure_packages(); print("依赖已就绪：", m.PINS, flush=True); return
    if not a.root: die("需要 --root")
    if a.cmd == "download": download(a.root)
    elif a.cmd == "smoke": smoke(a.root, a.weights)
    elif a.cmd == "run": run(a.root, a.weights, [int(x) for x in a.folds.split(",")], False)
    elif a.cmd == "full": full(a.root, a.weights)
    else: finalize(a.root, a.weights)


if __name__ == "__main__":
    main()
