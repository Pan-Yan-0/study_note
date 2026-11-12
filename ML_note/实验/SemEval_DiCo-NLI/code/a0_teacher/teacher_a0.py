"""A0：Q0 教师在 B2 生成行上的推理（Kaggle T4×2，fp16）。研究对齐 v4 / v4.1，冻结计划变更记录 11-12 条目。

  python3 teacher_a0.py setup                                   装固定版本的 transformers / peft / accelerate / datasets（上游 ensure_packages）
  python3 teacher_a0.py download --root R                       下载 Qwen3-4B-Instruct-2507 快照 cdbee75f（逐文件核对），写 R/model.json
  python3 teacher_a0.py run --root R --weights W [W ...] --folds 0,2,4 [--smoke]
                                                                环境变量 A0_GPU 指定这个进程用哪张卡；逐折、逐种子
  python3 teacher_a0.py finalize --root R                       核对 15 份回执、打包 R/download/a0_results.zip

每个 (fold, seed)：
  1. 身份：adapter_model.safetensors、adapter_config.json 的哈希等于归档 COMPLETE.json；选中轮等于 trained.json；
     adapter_config 的 r / alpha / target_modules / dropout 等于上游常量。
  2. 载入：和上游 train() 相同的路径（get_peft_model → cast_mixed_precision_params → set_peft_model_state_dict），
     载入后逐张量和文件比对。
  3. 内层验证重评：输入和顺序同上游（dict.fromkeys 去重），argmax 和归档 val_probs/epoch{best}.json 的一致率 ≥ 98%。
  4. seed 42 另做逐折复现：按归档 cache.jsonl 的行序（即历史的"按长度降序、每 64 条一组"）重算全部输入；
     留出行 sym 标签和归档 pred_fold0_sym.csv 的一致率 ≥ 99%，四类概率最大绝对差 ≤ 0.10。另报平均绝对差和 argmax 翻转数。
     本折 seed 42 没过门槛，这一折的三个种子都不出教师文件，进程以非零状态退出。
  5. 教师推理：B2 本折实际送训的生成行，两个方向去重（data/b2_inputs.json 的顺序），按两段长度之和降序（稳定排序）、
     每 64 条一组调用上游 score()。写 teacher/fold{k}/seed{s}.jsonl 和同名 .json 回执（含文件哈希）。
已完成且哈希对得上的任务跳过；回执在、文件哈希对不上时报错，不覆盖。
--smoke：只做 fold 列表里第一折的 seed 42，教师推理只取排序后的前 128 条，写在 R/smoke/ 下。
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "q4b"))
SEEDS = (42, 43, 44)
GATE = {"holdout_sym_agree": 0.99, "max_abs_diff": 0.10, "val_argmax_agree": 0.98}
CHUNK = 64
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
SWAP = [0, 2, 1, 3]
VERSION = "a0-teacher-v1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def rj(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def wj(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding="utf-8"); os.replace(t, p)


def die(msg, code=2):
    print("停止：" + msg, flush=True); raise SystemExit(code)


def code_identity():
    names = ["teacher_a0.py", "q4b/upstream_qwen.py", "q4b/common.py", "q4b/worker.py", "q4b/expected_data.json"]
    return {n: sha(HERE / n) for n in names}


def data_identity():
    d = HERE / "data"
    return {n: sha(d / n) for n in ("b2_inputs.json",)}


def archive(fold, seed):
    return HERE / "archive" / "Q0" / f"seed{seed}" / f"fold{fold}"


def argmax(p): return max(range(4), key=lambda k: p[k])


def symp(p, q): return [(p[k] + q[SWAP[k]]) / 2 for k in range(4)]


def sort_inputs(pairs):
    return sorted(pairs, key=lambda x: -len(x[0]) - len(x[1]))


def chunks(seq, n=CHUNK): return [seq[i:i + n] for i in range(0, len(seq), n)]


def score_all(teacher, pairs):
    out = []
    for part in chunks(pairs):
        res = teacher.score(part)
        if len(res) != len(part): die("score 返回的行数不对")
        for r in res:
            p = r["p"]
            if len(p) != 4 or abs(sum(p) - 1) > 1e-4 or min(p) < 0: die("概率非法")
        out.extend(res)
    return out


# ---------- 上游模块和模型 ----------
def upstream(root):
    import worker  # q4b/worker.py：load() 设置环境变量、加载上游模块、核对官方数据哈希
    data = HERE / "official"
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    gpu = os.environ.get("A0_GPU", "0")
    q = worker.load(Path(root), data, 0, False, 8)
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
    ids = rj(archive(fold, seed) / "actual_split.json")["val"]
    return [byid[i] for i in ids]


def recheck_val(teacher, fold, seed, best, byid):
    rows = val_rows(fold, seed, byid)
    inputs = list(dict.fromkeys((r["text1"], r["text2"]) for r in rows))
    old = rj(archive(fold, seed) / "val_probs" / f"epoch{best}.json")
    if len(old) != len(inputs): die("归档内层验证概率条数不对")
    new = [x["p"] for x in teacher.score(inputs)]
    agree = sum(argmax(p) == argmax(o) for p, o in zip(new, old)) / len(old)
    diff = max(abs(a - b) for p, o in zip(new, old) for a, b in zip(p, o))
    return {"rows": len(old), "argmax_agree": agree, "max_abs_diff": diff, "pass": agree >= GATE["val_argmax_agree"]}


def reproduce(teacher, fold, byid):
    a = archive(fold, 42)
    lines = [json.loads(l) for l in open(a / "cache.jsonl", encoding="utf-8") if l.strip()]
    pairs = [(x["a"], x["b"]) for x in lines]
    new = score_all(teacher, pairs)  # 按归档行序每 64 条一组，即历史分批
    diffs = [abs(p - o) for x, r in zip(lines, new) for p, o in zip(r["p"], x["p"])]
    flips = sum(argmax(r["p"]) != argmax(x["p"]) for x, r in zip(lines, new))
    prob = {(x["a"], x["b"]): r["p"] for x, r in zip(lines, new)}
    test = [byid[i] for i in rj(a / "actual_split.json")["test"]]
    arch = {r["instance_id"]: r["label"] for r in csv.DictReader(open(a / "pred_fold0_sym.csv", encoding="utf-8"))}
    lab = {r["instance_id"]: LABELS[argmax(symp(prob[(r["text1"], r["text2"])], prob[(r["text2"], r["text1"])]))] for r in test}
    if set(lab) != set(arch): die("留出行和归档预测对不上")
    agree = sum(lab[i] == arch[i] for i in lab) / len(lab)
    out = {"inputs": len(pairs), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs), "argmax_flips": flips,
           "holdout_rows": len(lab), "holdout_sym_agree": agree}
    out["pass"] = agree >= GATE["holdout_sym_agree"] and out["max_abs_diff"] <= GATE["max_abs_diff"]
    return out


def official_rows():
    p = HERE / "official" / "final_data" / "train" / "dico_nli_train_track1_participant_labeled.csv"
    return {r["instance_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def run_task(teacher, root, weights, fold, seed, byid, b2, smoke):
    base = Path(root) / ("smoke" if smoke else "teacher") / f"fold{fold}"
    out, rec = base / f"seed{seed}.jsonl", base / f"seed{seed}.json"
    if rec.exists():
        r = rj(rec)
        if r.get("status") != "pass": die(f"{rec} 记录为未通过；先看原因，不自动重跑")
        if not out.exists() or sha(out) != r["output_sha256"]: die(f"{out} 和回执的哈希不同；不覆盖，请检查")
        if r["code"] != code_identity() or r["data"] != data_identity(): die(f"{rec} 是别的代码或数据版本写的")
        print(f"跳过（已完成并核对）：fold{fold} seed{seed}", flush=True); return
    t0 = time.time()
    d, best, hashes = identity(weights, fold, seed)
    load = teacher.load(d)
    receipt = {"version": VERSION, "fold": fold, "seed": seed, "best_epoch": best, "adapter_sha256": hashes, "load": load,
               "env": teacher.env, "code": code_identity(), "data": data_identity(), "gate": GATE, "smoke": smoke}
    receipt["val_recheck"] = recheck_val(teacher, fold, seed, best, byid)
    if seed == 42:
        receipt["reproduction"] = reproduce(teacher, fold, byid)
    ok = receipt["val_recheck"]["pass"] and receipt.get("reproduction", {"pass": True})["pass"]
    if not ok:
        receipt["status"] = "fail"; wj(rec, receipt)
        die(f"fold{fold} seed{seed} 没过门槛：{receipt.get('reproduction')} {receipt['val_recheck']}", 3)
    pairs = sort_inputs([tuple(x) for x in b2[str(fold)]])
    if smoke: pairs = pairs[:128]
    t1 = time.time()
    res = score_all(teacher, pairs)
    tmp = out.with_suffix(".jsonl.tmp"); out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for (a, b), r in zip(pairs, res):
            f.write(json.dumps({"a": a, "b": b, "p": r["p"], "mass": r["mass"], "top_is_letter": r["top_is_letter"]}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    receipt.update(status="pass", inputs=len(pairs), output_sha256=sha(out),
                   seconds={"total": time.time() - t0, "teacher_inference": time.time() - t1},
                   mean_answer_mass=sum(r["mass"] for r in res) / len(res))
    wj(rec, receipt)
    print(f"完成：fold{fold} seed{seed}  内层验证一致 {receipt['val_recheck']['argmax_agree']:.3f}  "
          + (f"复现 sym 一致 {receipt['reproduction']['holdout_sym_agree']:.3f} 最大差 {receipt['reproduction']['max_abs_diff']:.3g}  " if seed == 42 else "")
          + f"教师 {len(pairs)} 条  用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)


def run(root, weights, folds, smoke, teacher_cls=Teacher):
    byid = official_rows()
    b2 = rj(HERE / "data" / "b2_inputs.json")["inputs"]
    teacher = teacher_cls(root)
    try:
        for fold in folds:
            for seed in (SEEDS[:1] if smoke else SEEDS):  # 每折先 seed 42（含复现门槛），过了才跑 43、44
                run_task(teacher, root, weights, fold, seed, byid, b2, smoke)
            if smoke: break
    finally:
        teacher.release()


def finalize(root):
    root = Path(root)
    problems = []
    for fold in range(5):
        for seed in SEEDS:
            rec = root / "teacher" / f"fold{fold}" / f"seed{seed}.json"
            if not rec.exists(): problems.append(f"缺 fold{fold} seed{seed}"); continue
            r = rj(rec); out = rec.with_suffix(".jsonl")
            if r.get("status") != "pass": problems.append(f"fold{fold} seed{seed} 未通过")
            elif not out.exists() or sha(out) != r["output_sha256"]: problems.append(f"fold{fold} seed{seed} 输出哈希不对")
            elif r["code"] != code_identity() or r["data"] != data_identity(): problems.append(f"fold{fold} seed{seed} 代码或数据版本不同")
            elif seed == 42 and not r["reproduction"]["pass"]: problems.append(f"fold{fold} 复现没过")
    summary = {"version": VERSION, "problems": problems, "code": code_identity(), "data": data_identity()}
    wj(root / "A0_SUMMARY.json", summary)
    if problems: die("还不能打包：" + "；".join(problems))
    import zipfile
    dest = root / "download"; dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest / "a0_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if p.is_file() and (rel.parts[0] in ("teacher", "smoke", "logs") or str(rel) in ("A0_SUMMARY.json", "model.json")):
                z.write(p, str(rel))
        for n in code_identity():
            z.write(HERE / n, "source/" + n)
    print("已打包：", dest / "a0_results.zip", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["setup", "download", "run", "finalize"])
    p.add_argument("--root"); p.add_argument("--weights", nargs="*", default=[]); p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.cmd == "setup":
        import importlib.util as u
        spec = u.spec_from_file_location("up", HERE / "q4b" / "upstream_qwen.py"); m = u.module_from_spec(spec); spec.loader.exec_module(m)
        m.ensure_packages(); print("依赖已就绪：", m.PINS, flush=True); return
    if not a.root: die("需要 --root")
    if a.cmd == "download": download(a.root)
    elif a.cmd == "run": run(a.root, a.weights, [int(x) for x in a.folds.split(",")], a.smoke)
    else: finalize(a.root)


if __name__ == "__main__":
    main()
