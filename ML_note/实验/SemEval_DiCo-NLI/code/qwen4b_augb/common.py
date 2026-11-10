"""CPU-only identity, data and artifact checks for the Qwen3-4B augmentation queue (built on qwen8b_v5.5)."""
import csv, hashlib, json, math, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
UPSTREAM = '502e638814ba2b7a840f33aa3b481de1f1e0bf2887ff7d976dd86131d3b603b0'
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REV = 'cdbee75f17c01a7cc42f958dc650907174af0554'   # 和 JOINT-QWEN、第四到第六组相同的快照
VERSION = 'qwen4b_augb-v1'
GROUPS = ('Q0', 'Q-B1', 'Q-step1')
STAGES = {1: ('Q0', 'Q-B1'), 2: ('Q-step1',)}
SOURCES = ('common.py','worker.py','queue_augb.py','upstream_qwen.py','run_a10.sh','compat.py','expected_data.json')
# 实验 B +1 的生成数据（P1C-R 放行，清单 b380b4a6…）、逐折选用、五折划分；和 mDeBERTa 的 B1-Haiku 是同一批
AUGB_DATA = {'augb_data/manifest.csv': 'b380b4a6a836e7056b6f30405a48ed1d9ce66852b78f0be4f8ccf201769e0882',
             'augb_data/fold_selection.csv': '6f289cbac76e62143660cf849e61fe3e167b16b1e6dd1d18f635c7c1db0066a4',
             'augb_data/split_manifest.json': 'f9363290af2854427c4b41127d2c305db435e99d8d5f1c84c5990e1672e77564'}
# 交接和打包用的同一份白名单（prepare_upload 和 pack 共用）；SOURCES 之外的不进代码身份
SHIP = SOURCES + ('README.md','test_queue.py','prepare_upload.py','a10_session.sh','validation.txt')
BASELINE = {'baseline/zero4b_cache_a10_t1.jsonl':'a4bff3d86fac722ada51bb7632577c033da51d88a6da6b762921a18969b40c2e'}  # 仓库 metrics/qwen_sft_a10_t1/track1/zero_shot/cache.jsonl 的副本
LOCK_NAMES = ('.queue.lock','.session.lock')
PREFIX = 'qwen4b_augb'
def require(ok, message):
    if not ok: raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,indent=2)); os.replace(t,p)
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def pipeline():
    require(sha(HERE/'upstream_qwen.py')==UPSTREAM,'上游核心字节变化')
    for n,h in AUGB_DATA.items(): require((HERE/n).is_file() and sha(HERE/n)==h,'实验 B 的数据文件缺失或被改动: '+n)
    return digest({**{n:sha(HERE/n) for n in SOURCES},**AUGB_DATA})
def bind(p,identity):
    if Path(p).exists(): require(read(p)==identity,'来源或配置变化，必须使用新输出目录')
    else: write(p,identity)
def valid_p(p):
    return len(p)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<1e-5
def seal(d,identity):
    d=Path(d)
    files={str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name!='COMPLETE.json' and 'checkpoints' not in p.parts and 'checkpoints_invalid' not in p.parts}
    require('done.json' in files and any(n.endswith('adapter_model.safetensors') for n in files),'结果或权重不完整')
    write(d/'COMPLETE.json',{'identity':identity,'files':files})
def verify(d,identity):
    d=Path(d); m=read(d/'COMPLETE.json'); require(m['identity']==identity,'完成回执身份变化')
    require('done.json' in m['files'] and any(n.endswith('adapter_model.safetensors') for n in m['files']),'回执缺少必要产物')
    for n,h in m['files'].items():
        p=d/n; require(p.resolve().is_relative_to(d.resolve()),'非法回执路径')
        require(p.is_file() and sha(p)==h,'完成文件缺失/变化: '+n)
def tasks(stage):
    """先同种子的五折；同一折里 Q0 和 Q-B1 相邻，中途停下时已完成的部分仍是成对的。"""
    return [(g,f,s) for s in (42,43,44) for f in range(5) for g in STAGES[stage]]
def task_dir(root,name,group,fold,seed): return Path(root)/name/group/f'seed{seed}'/f'fold{fold}'

# ---------- 训练数据（只依赖标准库，CPU 测试直接调用） ----------
def augb_split(fold):
    return {k:set(v) for k,v in read(HERE/'augb_data/split_manifest.json')['folds'][str(fold)].items()}
def augb_generated(fold):
    """该折选用的 B1 生成行，按 fold_selection.csv 的顺序；每行转成上游训练用的字段。"""
    with open(HERE/'augb_data/manifest.csv',encoding='utf-8') as f: man={r['sample_id']:r for r in csv.DictReader(f)}
    with open(HERE/'augb_data/fold_selection.csv',encoding='utf-8') as f: sel=list(csv.DictReader(f))
    out=[]
    for s in sel:
        if int(s['fold'])!=fold: continue
        r=man[s['sample_id']]; require(r['pair_uid']==s['pair_uid'],'选用表和清单的 pair_uid 不一致')
        out.append({'instance_id':'gen:'+r['sample_id'],'pair_id':'gen:'+r['pair_uid'],'text1':r['text1'],'text2':r['text2'],'label':r['label'],
                    '_usable_folds':r['usable_folds'],'_context_parent_ids':r['context_parent_ids'],'_source':r['source'],'_generator':r['generator_model']})
    return out
def build_fit(group,fit,generated):
    """Q0 = fit；Q-B1 = fit + 该折生成行；Q-step1 = fit + fit 再复制一遍（和 mDeBERTa 的 step+1 相同，行数等于 Q-B1）。"""
    require(group in GROUPS,'未知组 '+str(group))
    if group=='Q0': return list(fit)
    if group=='Q-step1': return list(fit)+[{**r,'instance_id':r['instance_id']+'#dup1'} for r in fit]
    return list(fit)+[{k:v for k,v in r.items() if not k.startswith('_')} for r in generated]
def check_generated(fold,fit,generated,full):
    """隔离检查：生成行在本折可用，上下文父样本都在本折 fit 里；full=True 时行数等于 fit（+1）。"""
    ids={r['instance_id'] for r in fit}
    for r in generated:
        require(str(fold) in set(r['_usable_folds'].split(';')),'生成行在本折不可用: '+r['instance_id'])
        parents=set(x for x in r['_context_parent_ids'].split(';') if x)
        require(parents and parents<=ids,'生成行的上下文父样本不在本折 fit: '+r['instance_id'])
        require(r['_source']=='claude-haiku-5-5' and r['_generator']=='haiku','生成来源不对: '+r['instance_id'])
        require(r['label'] in ('EQUIVALENCE','FORWARD_ENTAILMENT','BACKWARD_ENTAILMENT','NEGATIVE_OTHER'),'标签非法')
    if full: require(len(generated)==len(fit),'+1 的生成行数应等于 fit 行数: %d vs %d'%(len(generated),len(fit)))
def fit_digest(rows): return digest([[r['text1'],r['text2'],r['label']] for r in rows])

def ship_files():
    """白名单里实际存在的文件（相对 HERE），随包缓存和数据必须和登记的哈希一致。"""
    out = [n for n in SHIP if (HERE/n).is_file()]
    for n, h in {**BASELINE, **AUGB_DATA}.items():
        require((HERE/n).is_file() and sha(HERE/n) == h, '随包文件缺失或被改动: '+n); out.append(n)
    return out
def root_lock(root):
    """同一个 ROOT 同时只允许一个协调器写入。非阻塞：已有进程持有时立即报错。返回的文件对象要一直持有。"""
    import fcntl
    p = Path(root)/'.queue.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, 'a+')
    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close(); raise ValueError('这个 ROOT 已有任务正在运行（%s 被占用）。用 status 看进度，不要重复启动。' % p)
    return f
