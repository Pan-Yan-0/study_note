"""CPU coordinator for the Qwen3-4B augmentation queue.
阶段 1：Q0 + Q-B1（30 项）；阶段 2：Q-step1（15 项），只有阶段 1 全部完成、并显式给出 --confirm-stage2 才会开始。"""
import argparse, os, signal, subprocess, sys, time, zipfile
from common import *
def child(a,kind,group='Q0',fold=0,seed=42):
    cmd=[sys.executable,str(HERE/'worker.py'),kind,'--root',str(a.root),'--data',str(a.data),'--batch',str(a.batch),'--group',group,'--fold',str(fold),'--seed',str(seed)]
    log=a.root/'logs'/f'{kind}_{group}_f{fold}_s{seed}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    print('运行',kind,group,fold,seed,'日志',log,flush=True)
    with log.open('a') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try: return p.wait(timeout=7200 if kind=='download' else None)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM); p.wait(); raise RuntimeError('下载超过2小时，保留缓存并停止（重新运行会接着下载）')

SMOKE_DIRS=(('probe','Q-B1',42),('smoke','Q-B1',42),('smoke','Q0',43))
def actual_fit(d):
    """实际训练的行：task.json 记的是裁剪 / 补行之前的候选（探测是该组完整训练集，SMOKE 之后还会补最长的原始行），
    实际训练集以 actual_split.json 为准，并和 done.json 的 fit_rows 对上。"""
    ids=read(d/'actual_split.json')['fit']; done=read(d/'done.json')
    require(len(ids)==len(set(ids))==done['fit_rows'],'actual_split.json 的 fit 和 done.json 的 fit_rows 对不上：%s'%d)
    return ids
def probe_check(root):
    probe=task_dir(root,'probe','Q-B1',0,42); verify(probe,read(probe/'task.json'))
    require(len(actual_fit(probe))==16,'最长输入探测实际训练的不是 16 行')
def smoke_check(root):
    identity=read(root/'identity.json'); require(identity['pipeline']==pipeline(),'旧代码回执')
    probe_check(root)
    ds=[]
    for name,group,seed in SMOKE_DIRS[1:]:
        d=task_dir(root,name,group,0,seed); t=read(d/'task.json'); verify(d,t); ds.append(read(d/'done.json'))
        ids=actual_fit(d)
        if group=='Q-B1': require(any(i.startswith('gen:') for i in ids),'Q-B1 的 SMOKE 没有用到生成数据')
        else: require(not any(i.startswith('gen:') or '#dup' in i for i in ids),'Q0 的 SMOKE 混进了生成或复制的行')
    require(ds[0]['resumed_from_epoch']==1,'恢复验收缺失')
    require(any(x['checkpoint']=='checkpoint-999999' for x in ds[0]['resume']['skipped']),'坏检查点验收缺失')
    for d in ds:
        require(d['load_memory']['before_load']['allocated_gib']<0.5,'新进程显存未释放')
        require({c['epoch'] for c in d['adapter_checks']}=={h['epoch'] for h in d['history']},'缺少逐轮权重检查')
        require(all(c['val_argmax_agreement']>=0.98 for c in d['adapter_checks']),'权重重载预测不一致')
    return {'identity':identity,'runtime':read(root/'runtime.json'),'smoke_passed':True,'compatibility':'separate COMPATIBLE.json required','formal_ready':False}

def compat_check(root):
    c=read(root/'COMPATIBLE.json'); require(c['passed'] and c['identity']['pipeline']==pipeline() and c['runtime']==read(root/'runtime.json'),'兼容性回执失效')
    for n,h in c['files'].items(): require(sha(root/n)==h,'兼容性产物变化')
    return c

def stage_done(root,stage):
    n=0
    for g,f,s in tasks(stage):
        d=task_dir(root,'formal',g,f,s)
        if (d/'COMPLETE.json').exists():
            try: verify(d,read(d/'task.json')); n+=1
            except Exception: pass
    return n

def receipts(root):
    """只读：用正式入口同样的校验函数检查两个回执，不写任何文件。"""
    out={}
    for name,fn in (('SMOKE',smoke_check),('COMPATIBLE',compat_check)):
        if not (root/(name+'.json')).exists(): out[name]='还没有'; continue
        try: fn(root); out[name]='通过（与正式入口同一套校验）'
        except Exception as e: out[name]='没通过：%s: %s'%(type(e).__name__,e)
    for st in STAGES: out[f'stage{st}_complete']='%d / %d'%(stage_done(root,st),len(tasks(st)))
    return out

def pack(root):
    dest=root/'download'; dest.mkdir(exist_ok=True)
    for kind in ('results','weights','resume'):
        target=dest/f'{PREFIX}_{kind}.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            manifest={}
            for p in sorted(root.rglob('*')):
                if not p.is_file(): continue
                rel=p.relative_to(root); parts=rel.parts
                if parts[0] in ('download','ms_cache','hf_cache') or p.name=='model.json' or p.name in LOCK_NAMES: continue
                checkpoints=any(x in ('checkpoints','checkpoints_invalid') for x in parts)
                weights='adapters' in parts
                if (kind=='results' and (weights or checkpoints)) or (kind=='weights' and not weights) or (kind=='resume' and not checkpoints): continue
                z.write(p,str(rel)); manifest[str(rel)]=sha(p)
            for name in ship_files():
                p=HERE/name; z.write(p,'source/'+name); manifest['source/'+name]=sha(p)
            z.writestr('MANIFEST.json',json.dumps({'pipeline':pipeline(),'kind':kind,'files':manifest},indent=2))
        print('已保存',target)

def merge_predictions(rows_by_fold,expected):
    merged={}
    for rows in rows_by_fold:
        for r in rows:
            key=r['instance_id']; require(key not in merged,'OOF重复 ID: '+key); merged[key]=r
    require(set(merged)==set(expected),'OOF覆盖不完整')
    return [merged[k] for k in expected]

def collect(a,stage):
    from worker import load
    q=load(a.root,a.data,0,False,a.batch)
    gold=q.read_rows(q.data_file(1,'train','participant_labeled')); expected=[r['instance_id'] for r in gold]
    require(len(expected)==3042 and len(set(expected))==3042,'官方英语行数异常')
    for group in STAGES[stage]:
        out=a.root/'oof'/group; out.mkdir(parents=True,exist_ok=True)
        for seed in (42,43,44):
            for mode in ('plain','sym'):
                chunks=[]
                for fold in range(5):
                    d=task_dir(a.root,'formal',group,fold,seed); task=read(d/'task.json'); verify(d,task)
                    require(task['pipeline']==pipeline() and task['group']==group,'OOF混入旧代码或别的组')
                    rows=q.read_rows(str(d/f'pred_fold0_{mode}.csv'))  # 上游固定文件名，内容是本折留出
                    require({r['instance_id'] for r in rows}==set(read(d/'actual_split.json')['test']),'折内覆盖错误')
                    chunks.append(rows)
                rows=merge_predictions(chunks,expected); pred=out/f'seed{seed}_{mode}.csv'; q.write_rows(str(pred),rows,['instance_id','label'])
                ref=q.reference_for(1,'train',gold,str(out),'train')
                q.official_score(ref,str(pred),str(out/f'seed{seed}_{mode}_scores'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['plan','receipts','download','smoke','formal','collect','pack','compatibility'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--batch',type=int,choices=[4,8],default=8)
    p.add_argument('--stage',type=int,choices=[1,2],default=1); p.add_argument('--confirm-formal',action='store_true'); p.add_argument('--confirm-stage2',action='store_true')
    p.add_argument('--adapter',type=Path); p.add_argument('--baseline-cache',type=Path); a=p.parse_args()
    a.root=a.root.resolve(); a.data=a.data.resolve(); a.root.mkdir(parents=True,exist_ok=True)
    pipeline()
    if a.command=='receipts': print(json.dumps(receipts(a.root),ensure_ascii=False,indent=2)); return
    if a.command=='plan': print(json.dumps({'version':VERSION,'stage1':tasks(1),'stage2':tasks(2),'formal_ready':'requires current SMOKE and COMPATIBLE receipts; stage 2 also requires stage 1 complete and --confirm-stage2'},ensure_ascii=False,indent=2)); return
    lock=root_lock(a.root)  # download / smoke / compatibility / formal / collect / pack 互斥
    if os.environ.get('DICO_TEST_HOLD_SECONDS'): time.sleep(float(os.environ['DICO_TEST_HOLD_SECONDS']))  # 只给 CPU 测试用
    if a.command=='download': require(child(a,'download')==0,'模型准备失败'); return
    if a.command=='pack': pack(a.root); return
    try:
        if a.command=='compatibility':
            require(a.adapter and a.baseline_cache,'请提供 --adapter 和 --baseline-cache')
            for kind in ('download','zero','old','new','compare'):
                cmd=[sys.executable,str(HERE/'compat.py'),kind,'--root',str(a.root),'--data',str(a.data),'--adapter',str(a.adapter),'--baseline-cache',str(a.baseline_cache)]
                subprocess.run(cmd,check=True,timeout=7200 if kind=='download' else None)
        elif a.command=='smoke':
            require(child(a,'probe','Q-B1')==0,'最长输入探测失败；仅真实OOM才考虑新root改batch4')
            d=task_dir(a.root,'smoke','Q-B1',0,42)
            if not (d/'COMPLETE.json').exists():
                if not (d/'INTERRUPTED.json').exists(): require(child(a,'interrupt','Q-B1')==75,'中断阶段失败，查看日志')
                require(child(a,'resume','Q-B1')==0,'恢复阶段失败')
            require(child(a,'smoke','Q0',seed=43)==0,'Q0 的 SMOKE 失败')
            write(a.root/'SMOKE.json',smoke_check(a.root))
            print('SMOKE通过；正式运行还需同版本COMPATIBLE.json。')
        elif a.command=='formal':
            require(a.confirm_formal,'正式运行必须 --confirm-formal')
            if a.stage==2:
                require(a.confirm_stage2,'阶段 2（Q-step1）要等阶段 1 的结果按事先规则通过、由 Pan-Yan-0 确认后再跑：加 --confirm-stage2')
                require(stage_done(a.root,1)==len(tasks(1)),'阶段 1 还没全部完成，不能开始阶段 2')
            smoke_check(a.root); compat_check(a.root)
            require(read(a.root/'identity.json')['batch']==a.batch,'微批变化，需要新验收')
            for g,fold,seed in tasks(a.stage): require(child(a,'formal',g,fold,seed)==0,'正式任务失败，修复后续跑')
            collect(a,a.stage)
        elif a.command=='collect': collect(a,a.stage)
    finally: pack(a.root)
if __name__=='__main__': main()
