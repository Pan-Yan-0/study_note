"""CPU-only identity and artifact checks for the reconstructed queue."""
import hashlib, json, math, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
UPSTREAM = '502e638814ba2b7a840f33aa3b481de1f1e0bf2887ff7d976dd86131d3b603b0'
MODEL = 'Qwen/Qwen3-8B'
REV = 'b968826d9c46dd6066d109eabc6255188de91218'
SOURCES = ('common.py','worker.py','a1_queue.py','upstream_qwen.py','run_a10.sh','compat.py','expected_data.json')
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
    return digest({n:sha(HERE/n) for n in SOURCES})
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
def tasks(): return [(f,s) for s in (42,43,44) for f in range(5)]
