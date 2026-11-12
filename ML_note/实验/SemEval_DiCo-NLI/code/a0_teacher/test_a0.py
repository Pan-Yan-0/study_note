"""A0 包的 CPU 测试（不需要 torch）：在组装好的包目录里运行  python3 -I test_a0.py
假教师：对归档里有的输入返回归档概率，其余返回由文本确定的伪概率；真实 adapter 的载入和数值只能在 Kaggle SMOKE 里测。"""
import csv, functools, hashlib, json, random, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import teacher_a0 as T  # noqa: E402

PASS = []


def case(f):
    PASS.append(f); return f


def h(s): return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class Fake:
    perturb = 0.0  # 复现时把这么多比例的输入改成另一类

    def __init__(self, root):
        self.env = {"gpu": "fake", "dtype": "torch.float16"}; self.cur = None

    def load(self, d):
        parts = Path(d).parts
        self.cur = (int(parts[-3][4:]), int(parts[-4][4:]))  # fold, seed
        fold, seed = self.cur
        a = T.archive(fold, seed); best = T.rj(a / "trained.json")["best_epoch"]
        byid = T.official_rows()
        rows = T.val_rows(fold, seed, byid)
        inputs = list(dict.fromkeys((r["text1"], r["text2"]) for r in rows))
        self.known = dict(zip(inputs, T.rj(a / "val_probs" / f"epoch{best}.json")))
        if seed == 42:
            for l in open(a / "cache.jsonl", encoding="utf-8"):
                x = json.loads(l); self.known[(x["a"], x["b"])] = x["p"]
        return {"tensors": 504, "dtype": "torch.float16"}

    def score(self, pairs):
        out = []
        for a, b in pairs:
            p = self.known.get((a, b))
            if p is None:
                v = [1 + h(a + "|" + b + str(k)) % 7 for k in range(4)]; p = [x / sum(v) for x in v]
            elif self.perturb and h(a + b) % 1000 < 1000 * self.perturb:
                k = T.argmax(p); p = [0.0] * 4; p[(k + 1) % 4] = 1.0
            out.append({"p": list(p), "mass": 0.999, "top_is_letter": True})
        return out

    def release(self): pass


def fake_identity(weights, fold, seed):
    best = T.rj(T.archive(fold, seed) / "trained.json")["best_epoch"]
    return Path(f"/w/formal/Q0/seed{seed}/fold{fold}/adapters/epoch{best}"), best, {"adapter_model.safetensors": "x"}


def with_fake(fn):
    @functools.wraps(fn)
    def run():
        old = T.identity; T.identity = fake_identity
        try: fn()
        finally: T.identity = old; Fake.perturb = 0.0
    return run


@case
def bundle_integrity():
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        hsh, name = line.split("  ", 1)
        assert T.sha(HERE / name) == hsh, name
    b2 = T.rj(HERE / "data" / "b2_inputs.json")
    assert b2["rows_per_fold"] == {"0": 4380, "1": 4384, "2": 4372, "3": 4400, "4": 4400}
    assert sum(len(v) for v in b2["inputs"].values()) == 25372 and sum(len(v) for v in b2["inputs"].values()) * 3 == 76116


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
    root = Path(tempfile.mkdtemp())
    T.run(root, [], [0, 1], False, Fake)
    for fold in (0, 1):
        for seed in T.SEEDS:
            r = T.rj(root / "teacher" / f"fold{fold}" / f"seed{seed}.json")
            assert r["status"] == "pass" and r["val_recheck"]["argmax_agree"] == 1.0
            n = sum(1 for _ in open(root / "teacher" / f"fold{fold}" / f"seed{seed}.jsonl"))
            assert n == len(T.rj(HERE / "data/b2_inputs.json")["inputs"][str(fold)])
        rep = T.rj(root / "teacher" / f"fold{fold}" / "seed42.json")["reproduction"]
        assert rep["max_abs_diff"] == 0 and rep["holdout_sym_agree"] == 1.0 and rep["pass"]
    T.run(root, [], [0, 1], False, Fake)  # 全部跳过
    out = root / "teacher" / "fold1" / "seed43.jsonl"
    out.write_text(out.read_text() + "\n")
    try: T.run(root, [], [1], False, Fake); raise AssertionError("改过的输出应当报错")
    except SystemExit as e: assert e.code == 2
    shutil.rmtree(root)


@case
@with_fake
def output_order_is_sorted_and_complete():
    root = Path(tempfile.mkdtemp())
    T.run(root, [], [2], False, Fake)
    got = [(x["a"], x["b"]) for x in map(json.loads, open(root / "teacher/fold2/seed44.jsonl", encoding="utf-8"))]
    want = T.sort_inputs([tuple(x) for x in T.rj(HERE / "data/b2_inputs.json")["inputs"]["2"]])
    assert got == want and len(set(got)) == len(got)
    shutil.rmtree(root)


@case
@with_fake
def reproduction_gate_stops_fold():
    root = Path(tempfile.mkdtemp())
    Fake.perturb = 0.05  # 5% 的输入换类，留出 sym 一致率会低于 99%
    try: T.run(root, [], [3], False, Fake); raise AssertionError("复现门槛应当拦下")
    except SystemExit as e: assert e.code == 3
    r = T.rj(root / "teacher/fold3/seed42.json")
    assert r["status"] == "fail" and not r["reproduction"]["pass"]
    assert not (root / "teacher/fold3/seed42.jsonl").exists() and not (root / "teacher/fold3/seed43.json").exists()
    Fake.perturb = 0.0
    try: T.run(root, [], [3], False, Fake); raise AssertionError("失败回执应当阻止自动重跑")
    except SystemExit as e: assert e.code == 2
    shutil.rmtree(root)


@case
@with_fake
def smoke_writes_128_to_smoke_dir():
    root = Path(tempfile.mkdtemp())
    T.run(root, [], [4, 0], True, Fake)
    r = T.rj(root / "smoke/fold4/seed42.json")
    assert r["smoke"] and r["inputs"] == 128 and r["reproduction"]["pass"]
    assert not (root / "teacher").exists() and not (root / "smoke/fold0").exists()
    shutil.rmtree(root)


@case
@with_fake
def finalize_requires_all_fifteen():
    root = Path(tempfile.mkdtemp())
    T.run(root, [], [0, 1, 2, 3], False, Fake)
    try: T.finalize(root); raise AssertionError("缺 fold 4 时不应打包")
    except SystemExit as e: assert e.code == 2
    assert len(T.rj(root / "A0_SUMMARY.json")["problems"]) == 3
    T.run(root, [], [4], False, Fake)
    T.finalize(root)
    import zipfile
    names = zipfile.ZipFile(root / "download/a0_results.zip").namelist()
    assert sum(n.startswith("teacher/") and n.endswith(".jsonl") for n in names) == 15 and "A0_SUMMARY.json" in names
    assert any(n == "source/teacher_a0.py" for n in names)
    shutil.rmtree(root)


@case
def identity_rejects_wrong_adapter():
    w = Path(tempfile.mkdtemp())
    best = T.rj(T.archive(0, 43) / "trained.json")["best_epoch"]
    d = w / f"formal/Q0/seed43/fold0/adapters/epoch{best}"; d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"not the real adapter"); (d / "adapter_config.json").write_text("{}")
    try: T.identity([str(w)], 0, 43); raise AssertionError("哈希不同应当报错")
    except SystemExit as e: assert e.code == 2
    try: T.identity([str(w)], 0, 44); raise AssertionError("找不到应当报错")
    except SystemExit as e: assert e.code == 2
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
        f(); print("通过", f.__name__)
    print(len(PASS), "项全部通过")
