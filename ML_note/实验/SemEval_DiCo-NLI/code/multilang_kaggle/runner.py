"""Independent T4 workers, immutable completed tasks, portable result archives."""
import argparse, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REV='a0484667b22365f84929a935b5e50a51f71f159d'
UPSTREAM='83bfe561f0f5ab5f9ff52b37dd9619faf34d3ebcb7a34032a14fa55f9b597f63'

def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def identity(): return hashlib.sha256((sha(HERE/'runner.py')+sha(HERE/'upstream_mdeberta.py')+sha(HERE/'official_files.json')).encode()).hexdigest()
def tasks(): return [(t,s,f) for s in (42,43,44) for f in range(5) for t in (4,2,3)]
def tag(t): return 't%s_s%s_f%s'%tuple(t)
def target(root,t): return Path(root)/'formal'/tag(t)
def task_result(root,t,smoke=False):
    tr,s,f=t
    return Path(root)/('smoke' if smoke else 'formal')/tag(t)/f'track{tr}/ce/seed{s}/fold{f}'
def safe_extract(archive,dest):
    dest=Path(dest).resolve()
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'ZIP损坏')
        for n in z.namelist(): require((dest/n).resolve().is_relative_to(dest),'ZIP越界路径')
        z.extractall(dest)

def module(root,t=None,smoke=False,gpu=None):
    root=Path(root).resolve()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0,1')
    os.environ.update(DICO_WORK=str(root/'work'),DICO_DIR=str(root/'official'),DICO_STAGE='smoke' if smoke else 'stage2',DICO_PLATFORM='Kaggle',HF_HOME=str(root/'hf_cache'))
    os.environ.setdefault('HF_ENDPOINT','https://huggingface.co')
    require(sha(HERE/'upstream_mdeberta.py')==UPSTREAM,'原训练源码哈希不匹配')
    spec=importlib.util.spec_from_file_location('baseline_upstream',HERE/'upstream_mdeberta.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Upstream sets CUDA_VISIBLE_DEVICES=0 at import. Restore worker assignment BEFORE importing torch.
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu) if gpu is not None else visible
    m.OUT=str(root/('smoke' if smoke else 'formal')/tag(t)) if t else str(root/'aggregate')
    original=m.common_fingerprint
    m.common_fingerprint=lambda:{**original(),'multilang_driver':identity()}
    return m

def environment():
    import torch, importlib.metadata as md
    require(torch.cuda.is_available(),'没有GPU')
    require('T4' in torch.cuda.get_device_name(0),'本包冻结为T4，不混合其他GPU')
    return {'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
            'python':sys.version.split()[0],**{k:md.version(k) for k in ('transformers','numpy','sentencepiece','huggingface_hub')}}

def validate_task(root,t,m=None):
    d=task_result(root,t);m=m or module(root,t)
    done=read(d/'done.json');require(read(d/'task.json')==m.task_fingerprint(m.make_task(t[0],'ce',t[1],t[2],t[2]==0),REV),'任务指纹变化')
    require(done['code_sha256']==identity() and done['precision']=='fp16 autocast','任务代码/精度不符')
    require(done['env']['runtime']==read(Path(root)/'runtime.json'),'任务运行环境不同')
    require(len(read(d/'history.json'))==12,'未完成12轮')
    manifest=read(d/'verified.json')
    require(all((d/n).is_file() and sha(d/n)==h for n,h in manifest.items()),'任务文件缺失或变化')
    return done

def worker(root,t,gpu,smoke):
    m=module(root,t,smoke,gpu);env=environment()
    require(env==read(Path(root)/'runtime.json'),'环境与当前正式组不一致')
    import torch
    require(torch.cuda.device_count()==1,'每个训练进程必须只看到一张T4')
    print('物理GPU',gpu,'；可见卡数',torch.cuda.device_count(),'；卡型',torch.cuda.get_device_name(0),flush=True)
    # Freeze intended T4 precision instead of relying on version-dependent capability detection.
    m.amp_dtype=lambda: torch.float16
    print("固定训练精度：",m.amp_dtype(),flush=True)
    torch.set_num_threads(2)
    m._MODEL_PATH['path']=read(Path(root)/'model.json')['path']
    m.ENV.update({**m.env_info(),'platform':'Kaggle','amp':'fp16','runtime':env,'physical_gpu':gpu})
    m.check_common_fingerprint()
    task=m.make_task(t[0],'ce',t[1],t[2],t[2]==0)
    original=m.TRAIN_FN
    def strict(*a,**kw):
        result=original(*a,**kw)
        require(result.get('precision')=='fp16 autocast',f"训练精度不符：实际={result.get('precision')!r}，尝试记录={result.get('precision_attempts')!r}；要求fp16 autocast")
        return result
    m.TRAIN_FN=strict
    m.run_task(task,REV,identity())
    d=task_result(root,t,smoke)
    done=read(d/'done.json')
    require(done['precision']=='fp16 autocast','错误精度')
    if smoke:
        full=m.parts_for(t[0],t[2],True,smoke=False)
        timing=original(task,full,max_steps=12)
        require('failed' not in timing,'正式规模测速失败')
        step=sum(timing['step_times'][3:])/len(timing['step_times'][3:])
        inference=timing['infer_seconds_per_input']
        estimate=(math.ceil(len(full['fit'])/m.BATCH_SIZE)*12*step+12*(len(m.needed_inputs(full['val'],both=False))*inference+timing['clone_seconds'])+sum(len(m.needed_inputs(full[k])) for k in ('test','dev'))*inference+timing['load_seconds'])/60
        write(Path(root)/'timing'/f'track{t[0]}.json',{'minutes_estimate':estimate,'seconds_per_step':step,'timing':timing})
    write(d/'verified.json',{p.relative_to(d).as_posix():sha(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name!='verified.json'})

def prepare(root,resume):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    expected=read(HERE/'official_files.json')
    candidates=[p for p in (HERE/'official').iterdir() if p.is_dir() and (p/'evaluation_functions/__init__.py').is_file() and (p/'final_data').is_dir()]
    require(len(candidates)==1,'已解压官方数据目录不唯一')
    source=candidates[0]
    for n,h in expected.items():
        require((source/n).is_file() and sha(source/n)==h,'官方数据或评分文件不同: '+n)
    if not (root/'official').exists():shutil.copytree(source,root/'official')
    for n,h in expected.items():
        require((root/'official'/n).is_file() and sha(root/'official'/n)==h,'工作目录数据变化: '+n)
    if resume:
        with zipfile.ZipFile(resume) as z:
            meta=json.loads(z.read('MANIFEST.json'));require(meta['identity']==identity(),'恢复包代码/数据版本不同')
            require(z.testzip() is None,'恢复包损坏')
            for n,h in meta['files'].items():
                require((root/n).resolve().is_relative_to(root.resolve()),'恢复路径越界')
                b=z.read(n);require(hashlib.sha256(b).hexdigest()==h,'恢复文件哈希错误')
                # Only import completed tasks and frozen environment. Session clocks/SMOKE are not portable.
                if n.startswith('formal/') or n=='runtime.json':
                    p=root/n
                    if p.exists():require(p.read_bytes()==b,'已有文件与恢复包冲突')
                    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    m=module(root)
    env=environment()
    import torch
    require(torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'请选择T4×2')
    if (root/'runtime.json').exists():require(read(root/'runtime.json')==env,'恢复环境与原组不同，先核对依赖，不混入结果')
    write(root/'runtime.json',env)
    path,revision,source=m.fetch_model(REV);require(revision==REV,'模型版本变化')
    write(root/'model.json',{'path':path,'revision':revision,'source':source})
    for t in tasks():
        if (task_result(root,t)/'verified.json').exists():validate_task(root,t,module(root,t))
    print('准备完成；45项CE基线，T4×2独立任务。',flush=True)

def pack(root):
    root=Path(root);temp=root/'results.zip.tmp';files={}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for base in ('formal','smoke','timing','logs','aggregate'):
            for p in sorted((root/base).rglob('*')):
                if not p.is_file() or p.suffix=='.tmp':continue
                if base=='formal':
                    job=p.relative_to(root/'formal').parts[0]
                    if not list((root/'formal'/job).glob('track*/ce/seed*/fold*/verified.json')):continue
                n=p.relative_to(root).as_posix();b=p.read_bytes();z.writestr(n,b);files[n]=hashlib.sha256(b).hexdigest()
        for n in ('runtime.json','status.json','COMPLETE.json','smoke_ok.json'):
            if (root/n).exists():
                b=(root/n).read_bytes();z.writestr(n,b);files[n]=hashlib.sha256(b).hexdigest()
        z.writestr('MANIFEST.json',json.dumps({'identity':identity(),'files':files}))
    dest=root/'kaggle_multilang_results.zip';temp.replace(dest)
    print('已保存结果包：',dest,flush=True)

def collect(root):
    root=Path(root);m=module(root);out={};complete_tasks=[]
    for t in tasks():
        if not (task_result(root,t)/'verified.json').exists():continue
        validate_task(root,t,module(root,t));complete_tasks.append(t)
    m.OUT=str(root/'aggregate');fam,folds=m.split_base()
    for tr in (2,3,4):
        rows=m.read_rows(m.data_file(tr,'train','reference'))
        for seed in (42,43,44):
            if not all((tr,seed,f) in complete_tasks for f in range(5)):continue
            maps=[m.load_logits(str(task_result(root,(tr,seed,f))/'logits_test.json')) for f in range(5)]
            require(all(m.all_finite(x) for x in maps),'OOF存在非有限值')
            d=root/'aggregate'/f'track{tr}/seed{seed}';scores={}
            for mode in ('plain','sym'):
                labels=[m.argmax_label(m.probs(maps[folds[fam[r['pair_id']]]],r['text1'],r['text2'],mode)) for r in rows]
                scores[mode]=m.official_score(rows,labels,str(d),'oof_'+mode)
            out[f'track{tr}/seed{seed}']={'rows':len(rows),'scores':scores}
    write(root/'aggregate/summary.json',out)
    if len(complete_tasks)==45:
        require(len(out)==9,'OOF数量错误');write(root/'COMPLETE.json',{'tasks':45,'oof':out,'identity':identity()})
        print('COMPLETE：45项正式任务、9套完整OOF。',flush=True)
    return len(complete_tasks)

def enough_time(now,deadline,estimate): return now+max(600,estimate*60*1.6)<deadline

def queue(root,smoke,hours):
    root=Path(root)
    if not smoke:require(read(root/'smoke_ok.json')['identity']==identity(),'请先完成本次会话SMOKE')
    deadline=time.monotonic()+hours*3600
    pending=[(2,42,0),(3,42,0),(4,42,0)] if smoke else tasks()
    pending=[t for t in pending if smoke or not (task_result(root,t)/'verified.json').exists()]
    active={};failed=[];closed=False
    try:
        while pending or active:
            for gpu in (0,1):
                if gpu in active:continue
                if not pending or closed:continue
                # Complete earlier seed first; both GPUs may work on that seed only.
                seed=pending[0][1]
                if any(t[0][1]<seed for t in active.values()):continue
                t=pending[0]
                estimate=read(root/'timing'/f'track{t[0]}.json')['minutes_estimate'] if not smoke else 10
                if not enough_time(time.monotonic(),deadline,estimate):closed=True;pending=[];break
                pending.pop(0);log=root/'logs'/f'{"smoke" if smoke else "run"}_{tag(t)}.log';log.parent.mkdir(exist_ok=True)
                f=log.open('a',encoding='utf-8');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                p=subprocess.Popen([sys.executable,'-u',str(HERE/'runner.py'),'worker','--root',str(root),'--task',*map(str,t),'--gpu',str(gpu)]+(['--smoke'] if smoke else []),env=env,stdout=f,stderr=subprocess.STDOUT)
                active[gpu]=(t,p,f);print('开始',tag(t),'GPU',gpu,'日志',log,flush=True)
            for gpu,(t,p,f) in list(active.items()):
                if p.poll() is None:continue
                f.close();del active[gpu]
                if p.returncode:failed.append(tag(t));print('失败',tag(t),'请查看日志',flush=True)
                else:print('完成',tag(t),flush=True)
                pack(root)
            if active:time.sleep(3)
        if smoke:
            require(not failed and not closed,'SMOKE未全部完成；不允许正式训练')
            write(root/'smoke_ok.json',{'identity':identity(),'runtime':read(root/'runtime.json')})
            print('SMOKE通过，三个赛道测速：', {t:read(root/'timing'/f'track{t}.json')['minutes_estimate'] for t in (2,3,4)},flush=True)
        else:
            n=collect(root);write(root/'status.json',{'completed':n,'total':45,'failed':failed,'time_limit_reached':closed})
            print('本会话完成；累计',n,'/45；失败',failed,flush=True)
    finally:
        for t,p,f in active.values():
            p.terminate();p.wait();f.close()
        pack(root)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','run','worker','check','pack'])
    p.add_argument('--root',default='/kaggle/working/dico_multilang');p.add_argument('--resume')
    p.add_argument('--hours',type=float,default=10.5);p.add_argument('--task',type=int,nargs=3);p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.command=='worker':
        require(tuple(a.task) in tasks(),'非法任务');worker(root,tuple(a.task),a.gpu,a.smoke);return
    import fcntl
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare':prepare(root,a.resume)
        elif a.command in ('smoke','run'):queue(root,a.command=='smoke',a.hours)
        elif a.command=='check':collect(root);pack(root)
        else:pack(root)

if __name__=='__main__':main()
