"""Shared I/O, source identity and split/teacher validation. No training on import."""
import csv
import hashlib
import importlib.util
import json
import math
import os
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
QSHA = '502e638814ba2b7a840f33aa3b481de1f1e0bf2887ff7d976dd86131d3b603b0'
MSHA = '83bfe561f0f5ab5f9ff52b37dd9619faf34d3ebcb7a34032a14fa55f9b597f63'
QREV = 'cdbee75f17c01a7cc42f958dc650907174af0554'
MREV = 'a0484667b22365f84929a935b5e50a51f71f159d'
LABELS = ['EQUIVALENCE', 'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'NEGATIVE_OTHER']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def require(ok, message):
    if not ok: raise ValueError(message)

def portable_fp(fp):
    return {k: ({n.replace('\\','/'):h for n,h in v.items()} if k=='data_sha256' else v) for k,v in fp.items()}

def pipeline_id():
    return digest({n: sha(HERE/n) for n in ('joint_common.py','joint_worker.py','joint_queue.py','run_a10.sh','dico_qwen_methods.py','dico_mdeberta_methods.py')})

def rows(p):
    with open(p, encoding='utf-8', newline='') as f: return list(csv.DictReader(f))

def extract(zpath, dest):
    dest = Path(dest).resolve(); dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        require(z.testzip() is None, f'损坏的ZIP: {zpath}')
        for n in z.namelist():
            require((dest/n).resolve().is_relative_to(dest), 'ZIP含越界路径')
        z.extractall(dest)

def verify_bundle():
    for n, h in read(HERE/'SHA256.json').items(): require(sha(HERE/n) == h, f'上传文件损坏: {n}')

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def modules(root, smoke=False):
    root = Path(root).resolve()
    os.environ.update(DICO_WORK=str(root/'shared'), DICO_DIR=str(root/'official'),
                      DICO_SMOKE='1' if smoke else '0', DICO_STAGE='smoke' if smoke else 'stage2',
                      DICO_METHOD='ce', HF_ENDPOINT=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'),
                      HF_HOME=str(root/'hf_cache'), HF_HUB_DISABLE_XET='1',
                      DICO_PLATFORM='阿里云 PAI-DSW')
    require(sha(HERE/'dico_qwen_methods.py') == QSHA, 'Qwen训练源码发生变化')
    require(sha(HERE/'dico_mdeberta_methods.py') == MSHA, 'mDeBERTa训练源码发生变化')
    # Import both before creating CUDA contexts: upstream sources guard against late imports.
    q = load_module('joint_qwen', 'dico_qwen_methods.py')
    m = load_module('joint_mdeberta', 'dico_mdeberta_methods.py')
    return q, m

def signature(parts):
    keys = ('instance_id','pair_id','text1','text2','label')
    return {part: {'rows': len(parts[part]), 'sha256': digest([{k:r[k] for k in keys} for r in parts[part]])}
            for part in ('fit','val','test')}

def get_parts(q, m, fold, smoke=False):
    q.HOLDOUT_FOLD = fold
    qp = q.track_parts(1, q.make_split(), False)
    mp = m.parts_for(1, fold, True, smoke=False)
    require(signature(qp) == signature(mp), f'fold{fold}: Qwen与学生划分不一致')
    families = q.build_families(q.read_rows(q.data_file(1,'train','participant_labeled')))
    sets = [{families[r['pair_id']] for r in qp[k]} for k in ('fit','val','test')]
    require(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), '家族泄漏')
    if smoke:
        # Same small subset for teacher and student, unlike the two original smoke sizes.
        qp = {k: q.smoke_pairs(v, q.SMOKE_ROWS[k]) for k,v in qp.items()}
        ids = {k: {r['instance_id'] for r in v} for k,v in qp.items()}
        mp = {k: [r for r in v if r['instance_id'] in ids[k]] for k,v in mp.items()}
        require(signature(qp) == signature(mp), 'SMOKE划分不一致')
    return qp, mp

def qout(root, fold, smoke=False):
    return Path(root)/('smoke' if smoke else 'formal')/'qwen'/f'fold{fold}'/'dico_qwen_ce_results'

def tout(root, fold, seed, smoke=False):
    return Path(root)/('smoke' if smoke else 'formal')/'teachers'/f'fold{fold}'/f'seed{seed}'

def kdout(root, smoke=False):
    return Path(root)/('smoke' if smoke else 'formal')/'kd'/'dico_mdeberta_results'

def expected_inputs(q, parts):
    want = {}
    for part in ('fit','val'):
        for a,b in q.needed_inputs(parts[part]):
            k = q.key_of(a,b)
            require(k not in want or want[k] == part, '教师fit/val输入重叠')
            want[k] = part
    return want

def valid_p(p):
    return isinstance(p,list) and len(p)==4 and all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<1e-4

def teacher_rows(q, path, parts, complete=True):
    want = expected_inputs(q,parts); out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            x = json.loads(line); k = q.key_of(x['a'],x['b'])
            require(k not in out, '教师输入重复')
            require(k in want and want[k] == x['part'], '教师输入越界/分区错误')
            require(valid_p(x['p']), '教师概率无效')
            out[k] = x
    if complete: require(set(out)==set(want), '教师概率覆盖不完整')
    return out

def validate_teacher(q, root, fold, seed, parts, smoke=False):
    d=tout(root,fold,seed,smoke); meta=read(d/'identity.json'); done=read(d/'done.json')
    source=qout(root,fold,smoke); run=source/'track1'/f'seed{seed}'
    sd=read(run/'done.json'); fp=read(source/'fingerprint.json')
    require(meta['fold']==fold and meta['seed']==seed and meta['smoke']==smoke, '教师fold/seed/SMOKE错误')
    require(meta['pipeline_id']==pipeline_id() and meta['labels']==LABELS, '教师代码/标签顺序错误')
    require(meta['source_training_sha256']==QSHA, '教师训练代码来源错误')
    require(meta['split']==signature(parts), '教师数据划分错误')
    require(meta['source_fingerprint_sha256']==sha(source/'fingerprint.json'), '教师来源指纹变化')
    require(fp['model_revision']==QREV and fp['split']['holdout_fold']==fold and fp['smoke']==smoke and fp['method']=='ce', '教师来源设置错误')
    require(meta['best_epoch']==sd['best_epoch'], '教师轮次不匹配')
    require(meta['source_done_sha256']==sha(run/'done.json'), '教师来源SFT记录变化')
    adapter=run/'adapters'/f"epoch{sd['best_epoch']}"/'adapter_model.safetensors'
    require(meta['adapter_sha256']==sha(adapter), '教师权重变化')
    require(done['identity_sha256']==sha(d/'identity.json') and done['teacher_sha256']==sha(d/'teacher.jsonl'), '教师文件校验失败')
    require(done['checks']['val_argmax_agreement']>=0.98 and done['checks']['cache_argmax_agreement']>=0.98, '教师预测复核失败')
    by=teacher_rows(q,d/'teacher.jsonl',parts)
    return [by[q.key_of(r['text1'],r['text2'])]['p'] for r in parts['fit']], done['teacher_sha256']

def cache_fetch(mod, root, model_key, revision):
    """Use only locally rehashed, previously remote-verified snapshots."""
    original=mod.fetch_model
    manifest=Path(root)/'model_manifests'/f'{model_key}.json'
    def fetch(want):
        want=want or revision
        require(want==revision, '模型快照不匹配')
        if manifest.exists():
            x=read(manifest); p=Path(x['path'])
            require(x['revision']==revision, '缓存快照不匹配')
            if all((p/n).is_file() for n in x['files']):
                require(all(sha(p/n)==h for n,h in x['files'].items()), '模型缓存损坏')
                return str(p), revision, '本地已校验快照（重新核对SHA256）'
        path,rev,source=original(want)
        require(rev==revision,'下载快照不匹配')
        files={p.relative_to(path).as_posix():sha(p) for p in Path(path).rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(path).parts)}
        require(files,'模型快照为空')
        write(manifest,{'path':str(Path(path).resolve()),'revision':rev,'files':files,'source':source})
        return path,rev,source
    mod.fetch_model=fetch
