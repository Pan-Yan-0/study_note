"""A0 包的 CPU 测试（不需要 torch）：在组装好的包目录里运行  python3 -I test_a0.py
假教师：对归档里有的输入返回归档概率，其余返回由文本确定的伪概率；真实 adapter 的载入和数值只能在 Kaggle SMOKE 里测。"""
import csv, functools, hashlib, json, math, shutil, sys, tempfile, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import teacher_a0 as T  # noqa: E402

PASS = []
ENV = {"gpu": "Tesla T4", "torch": "2.x", "cuda": "12.x", "dtype": "torch.float16", "model_revision": T.REV,
       "transformers": "5.19.0", "peft": "0.21.2", "accelerate": "1.15.0", "datasets": "5.1.0"}
ADAPTER = {}  # (fold, seed) -> 假的 adapter 哈希；测试改它来模拟换了权重或归档


def case(f):
    PASS.append(f); return f


def h(s): return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class Fake:
    perturb = 0.0     # 复现时把这么多比例的输入改成另一类
    nan = False       # 教师推理时注入 NaN
    short_val = False  # 内层验证少返回一行
    modify = None      # 复现输入的概率改写函数 (a, b, p) -> p，模拟精度差异
    env = dict(ENV)

    def __init__(self, root):
        self.env = dict(Fake.env); self.cur = None; self.answer_tokens = {"option_ids": [[1], [2], [3], [4]]}

    def load(self, d):
        parts = Path(d).parts
        fold, seed = int(parts[-3][4:]), int(parts[-4][4:])
        a = T.archive(fold, seed); best = T.rj(a / "trained.json")["best_epoch"]
        byid = T.official_rows()
        self.val = T.val_inputs(fold, seed, byid)
        self.known = dict(zip(self.val, T.rj(a / "val_probs" / f"epoch{best}.json")))
        self.cache = set()
        if seed == 42:
            for l in open(a / "cache.jsonl", encoding="utf-8"):
                x = json.loads(l); self.known[(x["a"], x["b"])] = x["p"]; self.cache.add((x["a"], x["b"]))
        return {"tensors": 504, "dtype": "torch.float16"}

    def score(self, pairs):
        out = []
        for a, b in pairs:
            p = self.known.get((a, b))
            if p is None:
                v = [1 + h(a + "|" + b + str(k)) % 7 for k in range(4)]; p = [x / sum(v) for x in v]
                if Fake.nan: p = [math.nan, 0.5, 0.25, 0.25]
            elif Fake.perturb and h(a + b) % 1000 < 1000 * Fake.perturb:
                k = T.argmax(p); p = [0.0] * 4; p[(k + 1) % 4] = 1.0
            elif Fake.modify and (a, b) in self.cache:
                p = Fake.modify(a, b, list(p))
            out.append({"p": list(p), "mass": 0.999, "top_is_letter": True})
        if Fake.short_val and pairs == self.val: out = out[:-1]
        return out

    def release(self): pass


def fake_identity(weights, fold, seed):
    best = T.rj(T.archive(fold, seed) / "trained.json")["best_epoch"]
    return (Path(f"/w/formal/Q0/seed{seed}/fold{fold}/adapters/epoch{best}"), best,
            ADAPTER.get((fold, seed), {"adapter_model.safetensors": "a", "adapter_config.json": "c"}))


def new_root():
    root = Path(tempfile.mkdtemp())
    T.wj(root / "model.json", {"path": "/models/qwen", "revision": T.REV, "source": "test", "files": {"config.json": "x"}})
    return root


def with_fake(fn):
    @functools.wraps(fn)
    def run():
        old = T.identity; T.identity = fake_identity
        try: fn()
        finally:
            T.identity = old; Fake.perturb = 0.0; Fake.nan = False; Fake.short_val = False; Fake.modify = None; Fake.env = dict(ENV); ADAPTER.clear()
    return run


def stopped(fn, code):
    try: fn()
    except SystemExit as e:
        assert e.code == code, f"退出码 {e.code}，应为 {code}"; return
    raise AssertionError("应当停下")


@case
def bundle_integrity():
    T.bundle_identity()
    b2 = T.rj(HERE / "data" / "b2_inputs.json")
    assert b2["rows_per_fold"] == {"0": 4380, "1": 4384, "2": 4372, "3": 4400, "4": 4400}
    assert sum(len(v) for v in b2["inputs"].values()) * 3 == 76116


@case
def archived_cache_order_is_historic_batching():
    """归档 cache.jsonl 的行序 = needed_inputs(test + dev) 按两段长度之和降序的稳定排序，所以按行序每 64 条一组就是历史分批。"""
    byid = T.official_rows()
    dev = list(csv.DictReader(open(HERE / "official/final_data/dev/dico_nli_dev_track1_participant_labeled.csv", encoding="utf-8")))
    for fold in range(5):
        a = T.archive(fold, 42)
        test = [byid[i] for i in T.rj(a / "actual_split.json")["test"]]
        need, seen = [], set()
        for r in test + dev:
            for x in ((r["text1"], r["text2"]), (r["text2"], r["text1"])):
                if x not in seen: seen.add(x); need.append(x)
        lines = [json.loads(l) for l in open(a / "cache.jsonl", encoding="utf-8")]
        assert [(x["a"], x["b"]) for x in lines] == T.sort_inputs(need), fold


@case
@with_fake
def full_flow_resume_and_tamper():
    root = new_root()
    T.run(root, [], [0, 1], False, Fake)
    for fold in (0, 1):
        for seed in T.SEEDS:
            r = T.rj(root / "teacher" / f"fold{fold}" / f"seed{seed}.json")
            assert r["status"] == "pass" and r["val_recheck"]["argmax_agree"] == 1.0 and r["val_recheck"]["returned"] == r["val_recheck"]["rows"]
        rep = T.rj(root / "teacher" / f"fold{fold}" / "seed42.json")["reproduction"]
        assert rep["max_abs_diff"] == 0 and rep["holdout_sym_agree"] == 1.0 and rep["pass"] and rep["argmax_flips"] == 0
    T.run(root, [], [0, 1], False, Fake)  # 全部跳过
    out = root / "teacher" / "fold1" / "seed43.jsonl"
    out.write_text(out.read_text() + "\n")
    stopped(lambda: T.run(root, [], [1], False, Fake), 2)
    shutil.rmtree(root)


@case
@with_fake
def output_order_is_sorted_and_complete():
    root = new_root()
    T.run(root, [], [2], False, Fake)
    got = [(x["a"], x["b"]) for x in map(json.loads, open(root / "teacher/fold2/seed44.jsonl", encoding="utf-8"))]
    want = T.sort_inputs([tuple(x) for x in T.rj(HERE / "data/b2_inputs.json")["inputs"]["2"]])
    assert got == want and len(set(got)) == len(got)
    shutil.rmtree(root)


@case
@with_fake
def reproduction_gate_stops_fold_with_diagnostics():
    root = new_root()
    Fake.perturb = 0.05  # 5% 的输入换类，留出 sym 一致率会低于 99%
    stopped(lambda: T.run(root, [], [3], False, Fake), 3)
    r = T.rj(root / "teacher/fold3/seed42.json")
    rp = r["reproduction"]
    assert r["status"] == "fail" and not rp["pass"] and rp["flipped"] and "worst" in rp and "abs_diff_quantiles" in rp and rp["answer_tokens"]
    assert not (root / "teacher/fold3/seed42.jsonl").exists() and not (root / "teacher/fold3/seed43.json").exists()
    Fake.perturb = 0.0
    stopped(lambda: T.run(root, [], [3], False, Fake), 2)  # 失败回执阻止自动重跑
    shutil.rmtree(root)


def move(p, amount):
    """把 amount 的概率从最大类挪到第二大类（不翻转时 amount 要小于间距的一半）。"""
    s = sorted(range(4), key=lambda k: -p[k]); a = min(amount, p[s[0]]); p[s[0]] -= a; p[s[1]] += a; return p


@case
@with_fake
def single_tail_row_passes_but_reports_max():
    """v1.2：只有一条差得多、argmax 不变时通过，最大差照常记录。"""
    root = new_root()
    lines = [json.loads(l) for l in open(T.archive(0, 42) / "cache.jsonl", encoding="utf-8")]
    gaps = sorted(((T.top2_gap(x["p"]), (x["a"], x["b"])) for x in lines), reverse=True)
    target = gaps[0][1]  # 间距最大的一条，挪 0.3 也不翻转
    Fake.modify = lambda a, b, p: move(p, 0.3) if (a, b) == target else p
    T.run(root, [], [0], True, Fake)
    rp = T.rj(root / "smoke/fold0/seed42.json")["reproduction"]
    assert rp["pass"] and abs(rp["max_abs_diff"] - 0.3) < 1e-9 and rp["argmax_flips"] == 0 and rp["abs_diff_quantiles"]["0.99"] == 0
    shutil.rmtree(root)


@case
@with_fake
def broad_shift_or_many_flips_stop():
    """v1.2：2% 的输入差 0.2（不翻转）→ 99% 分位超过 0.10，停；只在非留出输入上翻转 1.5% → 留出一致但翻转率超过 1%，停。"""
    a = T.archive(1, 42)
    byid = T.official_rows()
    hold = {x for i in T.rj(a / "actual_split.json")["test"] for x in ((byid[i]["text1"], byid[i]["text2"]), (byid[i]["text2"], byid[i]["text1"]))}
    lines = [json.loads(l) for l in open(a / "cache.jsonl", encoding="utf-8")]
    wide = [(x["a"], x["b"]) for x in lines if T.top2_gap(x["p"]) > 0.5]
    shift = set(wide[:int(0.02 * len(lines)) + 1])
    root = new_root()
    Fake.modify = lambda x, y, p: move(p, 0.2) if (x, y) in shift else p
    stopped(lambda: T.run(root, [], [1], False, Fake), 3)
    rp = T.rj(root / "teacher/fold1/seed42.json")["reproduction"]
    assert not rp["pass"] and rp["argmax_flips"] == 0 and rp["abs_diff_quantiles"]["0.99"] > 0.10
    shutil.rmtree(root)
    other = [(x["a"], x["b"]) for x in lines if (x["a"], x["b"]) not in hold]
    flip = set(other[:int(0.015 * len(lines)) + 1])
    root = new_root()
    Fake.modify = lambda x, y, p: move(p, 1.0) if (x, y) in flip else p
    stopped(lambda: T.run(root, [], [1], False, Fake), 3)
    rp = T.rj(root / "teacher/fold1/seed42.json")["reproduction"]
    assert not rp["pass"] and rp["holdout_sym_agree"] == 1.0 and rp["flip_rate"] > 0.01
    shutil.rmtree(root)


@case
@with_fake
def receipt_with_old_gate_fields_is_rejected():
    """篡改回执：删掉 flip_rate 或 99% 分位，验收不通过。"""
    root = new_root()
    T.run(root, [], [0], True, Fake)
    rec = root / "smoke/fold0/seed42.json"
    for drop in ("flip_rate", "abs_diff_quantiles"):
        r = T.rj(rec); del r["reproduction"][drop]
        out, _ = T.task_paths(root, 0, 42, True)
        _, best, hashes = fake_identity([], 0, 42)
        bad = T.accept(r, out, 0, 42, True, best, hashes, T.SMOKE_N, len(T.val_inputs(0, 42, T.official_rows())), r["run_identity"])
        assert any("复现" in b for b in bad), drop
    shutil.rmtree(root)


@case
@with_fake
def smoke_failure_packs_diagnostics():
    root = new_root()
    Fake.perturb = 0.05
    stopped(lambda: T.smoke(root, [], Fake), 3)
    names = zipfile.ZipFile(root / "download/a0_smoke_failed.zip").namelist()
    assert "smoke/fold0/seed42.json" in names and "source/teacher_a0.py" in names
    assert not (root / "download/a0_smoke.zip").exists()
    shutil.rmtree(root)


@case
@with_fake
def nan_in_teacher_output_stops_without_receipt():
    root = new_root()
    Fake.nan = True
    stopped(lambda: T.run(root, [], [0], False, Fake), 2)
    assert not (root / "teacher/fold0/seed42.json").exists() and not (root / "teacher/fold0/seed42.jsonl").exists()
    shutil.rmtree(root)


@case
@with_fake
def short_inner_validation_stops_without_receipt():
    root = new_root()
    Fake.short_val = True
    stopped(lambda: T.run(root, [], [1], False, Fake), 2)
    assert not (root / "teacher/fold1/seed42.json").exists()
    shutil.rmtree(root)


@case
def json_writes_refuse_nan():
    p = Path(tempfile.mkdtemp()) / "x.json"
    try: T.wj(p, {"x": math.nan}); raise AssertionError("NaN 应当写不出")
    except ValueError: pass
    assert not p.exists()


@case
@with_fake
def resume_rejects_replaced_adapter():
    root = new_root()
    T.run(root, [], [0], False, Fake)
    ADAPTER[(0, 43)] = {"adapter_model.safetensors": "换过的权重", "adapter_config.json": "c"}  # 权重和归档一起换了（彼此匹配）
    stopped(lambda: T.run(root, [], [0], False, Fake), 2)
    shutil.rmtree(root)


@case
@with_fake
def resume_rejects_other_environment():
    root = new_root()
    T.run(root, [], [0], False, Fake)
    Fake.env = dict(ENV, torch="另一个版本")
    stopped(lambda: T.run(root, [], [0], False, Fake), 2)
    shutil.rmtree(root)


@case
@with_fake
def smoke_writes_128_to_smoke_dir():
    root = new_root()
    T.run(root, [], [4, 0], True, Fake)
    r = T.rj(root / "smoke/fold4/seed42.json")
    assert r["smoke"] and r["inputs"] == T.SMOKE_N and r["reproduction"]["pass"]
    assert not (root / "teacher").exists() and not (root / "smoke/fold0").exists()
    shutil.rmtree(root)


@case
@with_fake
def finalize_requires_all_fifteen_and_one_environment():
    root = new_root()
    T.run(root, [], [0, 1, 2, 3], False, Fake)
    stopped(lambda: T.finalize(root, []), 2)
    assert len(T.rj(root / "A0_SUMMARY.json")["problems"]) == 3
    Fake.env = dict(ENV, gpu="Tesla P100")  # 最后一折换了卡
    T.run(root, [], [4], False, Fake)
    stopped(lambda: T.finalize(root, []), 2)
    assert any("不一致" in x for x in T.rj(root / "A0_SUMMARY.json")["problems"])
    for s in T.SEEDS:
        for ext in (".json", ".jsonl"): (root / f"teacher/fold4/seed{s}{ext}").unlink()
    Fake.env = dict(ENV)
    T.run(root, [], [4], False, Fake)
    T.finalize(root, [])
    names = zipfile.ZipFile(root / "download/a0_results.zip").namelist()
    assert sum(n.startswith("teacher/") and n.endswith(".jsonl") for n in names) == 15 and "A0_SUMMARY.json" in names
    assert "source/teacher_a0.py" in names
    shutil.rmtree(root)


class Proc:
    def __init__(self, code): self.returncode = code
    def poll(self): return self.returncode


@case
@with_fake
def full_only_finalizes_when_both_workers_succeed():
    root = new_root()
    codes = iter([0, 2])
    stopped(lambda: T.full(root, [], spawn=lambda *a, **k: Proc(next(codes)), poll_seconds=0), 4)
    assert (root / "A0_FAILED.json").exists() and (root / "download/a0_failed_diagnostics.zip").exists()
    assert not (root / "download/a0_results.zip").exists()
    T.run(root, [], [0, 1, 2, 3, 4], False, Fake)
    T.full(root, [], spawn=lambda *a, **k: Proc(0), poll_seconds=0)
    assert (root / "download/a0_results.zip").exists()
    assert not (root / "A0_FAILED.json").exists() and (root / "logs/A0_FAILED_previous_1.json").exists()
    names = zipfile.ZipFile(root / "download/a0_results.zip").namelist()
    assert "A0_FAILED.json" not in names and "logs/A0_FAILED_previous_1.json" in names
    shutil.rmtree(root)


@case
def identity_rejects_wrong_adapter():
    w = Path(tempfile.mkdtemp())
    best = T.rj(T.archive(0, 43) / "trained.json")["best_epoch"]
    d = w / f"formal/Q0/seed43/fold0/adapters/epoch{best}"; d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"not the real adapter"); (d / "adapter_config.json").write_text("{}")
    stopped(lambda: T.identity([str(w)], 0, 43), 2)
    stopped(lambda: T.identity([str(w)], 0, 44), 2)
    shutil.rmtree(w)


@case
def adapter_found_under_extracted_zip_subfolder():
    w = Path(tempfile.mkdtemp())
    d = w / "qwen4b_augb_weights" / "formal/Q0/seed42/fold2/adapters/epoch3"; d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"x")
    assert T.adapter_dir([str(w)], 2, 42, 3) == d
    shutil.rmtree(w)


if __name__ == "__main__":
    for f in PASS:
        f(); print("通过", f.__name__, flush=True)
    print(len(PASS), "项全部通过")
