"""生成 A0_Teacher_Kaggle.ipynb：python3 make_notebook.py <输出路径>"""
import json, sys

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# A0：Q0 教师在 B2 生成行上的推理（Kaggle T4×2，fp16）

研究对齐 v4 / v4.1。只做推理，不训练。

**准备**
- 加两个数据集：A0 代码包 `a0_teacher_bundle`；Qwen 4B +1 阶段 1 的权重包，里面有 `formal/Q0/seed*/fold*/adapters/`。
- Accelerator 选 GPU T4 x2，打开 Internet。

**流程**
1. `RUN_FULL = False`，从头运行：装依赖 → 下模型 → SMOKE（fold 0 seed 42：身份、内层验证、逐折复现、128 条教师推理）。把下载链接里的 `a0_smoke.zip` 发给 Claude。
2. Claude 核对后，把 `RUN_FULL` 改成 `True`，用 Save & Run All (Commit) 后台跑。两张卡并行，预计 1.3–1.5 小时。
3. 跑完把 `a0_results.zip` 发给 Claude。中途停了的话，把上次的 Output 加成数据集再跑一次，已完成的任务核对哈希后跳过。""")
code("""RUN_FULL = False          # SMOKE 发给 Claude 核对后改成 True，再用 Save & Run All (Commit)
WORK = '/kaggle/working'
ROOT = WORK + '/a0_run'""")
code("""import glob, os, shutil, subprocess, sys, time, zipfile, hashlib, json
from pathlib import Path
# 1. 找代码包（含 teacher_a0.py 和 data/b2_inputs.json 的目录）
cands = [Path(p).parent for p in glob.glob('/kaggle/input/**/teacher_a0.py', recursive=True) if (Path(p).parent / 'data/b2_inputs.json').is_file()]
assert len(cands) >= 1, '没找到 A0 代码包：请把 a0_teacher_bundle 加为数据集'
src = cands[0]; BUNDLE = Path(WORK) / 'a0'
if BUNDLE.exists(): shutil.rmtree(BUNDLE)
shutil.copytree(src, BUNDLE)
bad = []
for line in (BUNDLE / 'SHA256SUMS').read_text().splitlines():
    h, name = line.split('  ', 1)
    if hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() != h: bad.append(name)
assert not bad, f'代码包文件和清单不一致：{bad[:5]}'
print('代码包：', src, '→', BUNDLE, '，清单核对通过')
# 2. 找权重：已解压的目录，或者没解压的 zip
WEIGHTS = sorted({str(Path(p).parents[6]) for p in glob.glob('/kaggle/input/**/formal/Q0/seed*/fold*/adapters/epoch*/adapter_model.safetensors', recursive=True)})
if not WEIGHTS:
    for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if n.startswith('formal/Q0/') and '/adapters/' in n]
            if names:
                out = Path(WORK) / 'weights'; f.extractall(out, members=names); WEIGHTS = [str(out)]; print('解压权重：', z); break
assert WEIGHTS, '没找到 Q0 权重：请把 Qwen 4B +1 阶段 1 的权重包加为数据集'
n = len(glob.glob(WEIGHTS[0] + '/formal/Q0/seed*/fold*/adapters/epoch*/adapter_model.safetensors'))
print('权重目录：', WEIGHTS, '，找到', n, '份 Q0 adapter（应为 15）')
assert n == 15
# 3. 上次的结果（换会话续跑）：把上次 Output 里的 a0_run/teacher 拷回来
for p in glob.glob('/kaggle/input/**/a0_run/teacher', recursive=True):
    if not Path(ROOT, 'teacher').exists():
        shutil.copytree(p, Path(ROOT, 'teacher')); print('恢复了上次的结果：', p)""")
code("""def sh(args, env=None, log=None):
    e = dict(os.environ, **(env or {}))
    print('$', ' '.join(args), flush=True)
    r = subprocess.run(args, env=e, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-6000:])
    if log: Path(log).parent.mkdir(parents=True, exist_ok=True); Path(log).write_text(r.stdout)
    assert r.returncode == 0, f'失败（退出码 {r.returncode}），把上面的输出发给 Claude'
PY = [sys.executable, '-I', str(BUNDLE / 'teacher_a0.py')]
t0 = time.time()
sh(PY + ['setup'], log=ROOT + '/logs/setup.log')
sh(PY + ['download', '--root', ROOT], log=ROOT + '/logs/download.log')
print(f'装依赖和下模型用时 {(time.time() - t0) / 60:.1f} 分钟')""")
code("""# SMOKE：fold 0 seed 42 的身份、内层验证重评、逐折复现门槛、前 128 条教师推理
from IPython.display import FileLink, display
t0 = time.time()
sh(PY + ['run', '--root', ROOT, '--weights', *WEIGHTS, '--folds', '0', '--smoke'], env={'A0_GPU': '0'}, log=ROOT + '/logs/smoke.log')
r = json.loads(Path(ROOT, 'smoke/fold0/seed42.json').read_text())
print(json.dumps({k: r[k] for k in ('status', 'val_recheck', 'reproduction', 'load', 'env', 'seconds')}, ensure_ascii=False, indent=1))
print(f'SMOKE 用时 {(time.time() - t0) / 60:.1f} 分钟')
with zipfile.ZipFile(WORK + '/a0_smoke.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in Path(ROOT).rglob('*'):
        if p.is_file() and (p.parts[len(Path(ROOT).parts)] in ('smoke', 'logs') or p.name == 'model.json'): z.write(p, str(p.relative_to(ROOT)))
display(FileLink('a0_smoke.zip'))
print('已完成：SMOKE。请把 a0_smoke.zip 发给 Claude；核对后把 RUN_FULL 改成 True，用 Save & Run All (Commit) 跑全量。')""")
code("""# 全量：两张 T4 各一个进程（GPU0：fold 0、2、4；GPU1：fold 1、3），每折先 seed 42（含复现门槛）
if RUN_FULL:
    t0 = time.time(); procs = []
    for gpu, folds in (('0', '0,2,4'), ('1', '1,3')):
        log = open(f'{ROOT}/logs/worker{gpu}.log', 'a')
        procs.append((gpu, subprocess.Popen(PY + ['run', '--root', ROOT, '--weights', *WEIGHTS, '--folds', folds],
                                            env=dict(os.environ, A0_GPU=gpu), stdout=log, stderr=subprocess.STDOUT, text=True), log))
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(120)
        done = len(glob.glob(ROOT + '/teacher/fold*/seed*.jsonl'))
        print(f'{(time.time() - t0) / 60:.0f} 分钟：已完成 {done} / 15 个教师文件', flush=True)
    for gpu, p, log in procs:
        log.close()
        print(f'--- worker{gpu} 退出码 {p.returncode}，日志末尾 ---'); print(Path(f'{ROOT}/logs/worker{gpu}.log').read_text()[-3000:])
    print(f'全量用时 {(time.time() - t0) / 60:.1f} 分钟')
    sh(PY + ['finalize', '--root', ROOT])
    shutil.copy2(ROOT + '/download/a0_results.zip', WORK + '/a0_results.zip')
    display(FileLink('a0_results.zip'))
    print('已完成：全量。请把 a0_results.zip 发给 Claude。')
else:
    print('RUN_FULL = False：只跑了 SMOKE。')""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("写好", sys.argv[1])
