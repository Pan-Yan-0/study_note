"""Single-GPU resumable experiment queue. Commands never release a cloud instance."""
import argparse
import shutil
import subprocess
import sys
import time
from joint_common import *

METHODS=['ce','rdrop','dir_rdrop','scl','moco_m0.999']

def prepare(root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    verify_bundle()
    previous=[HERE/'dico_joint_results.zip',HERE/'dico_joint_weights.zip']
    if any(p.exists() for p in previous):
        require(all(p.exists() for p in previous),'换实例恢复需同时提供总结果包和总权重包')
        for archive in previous:
            with zipfile.ZipFile(archive) as z:
                manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))
                require(manifest['pipeline_id']==pipeline_id(),'恢复包属于另一版代码')
                for name,h in manifest['files'].items():
                    if name in ('SMOKE_OK.json','COMPLETE.json','session_clock.json','budget_estimate.json') or name.startswith('smoke/') or '/dico_mdeberta_smoke/' in name:
                        continue  # A new instance must run a fresh GPU smoke check.
                    p=(root/name).resolve(); require(p.is_relative_to(root),'恢复包越界路径')
                    data=z.read(name); require(hashlib.sha256(data).hexdigest()==h,'恢复包文件损坏')
                    if not p.exists(): p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
    if not (root/'official'/'final_data').exists():
        temp=root/'official_extract'; extract(HERE/'SemEval-2027-Task-2-DiCo-NLI-main.zip',temp)
        found=list(temp.rglob('final_data')); require(len(found)==1,'官方数据目录不唯一')
        shutil.move(str(found[0].parent),str(root/'official'))
    md=root/'mdeberta'; md.mkdir(exist_ok=True)
    if not (md/'dico_mdeberta_results').exists(): extract(HERE/'dico_mdeberta_results_20261022-2320.zip',md)
    if not (md/'dico_qwen_teacher_results').exists(): shutil.copytree(HERE/'dico_qwen_teacher_results',md/'dico_qwen_teacher_results')
    dst=qout(root,0); weights=HERE/'dico_qwen_ce_weights_20261023-1017.zip'; results=HERE/'dico_qwen_ce_results_20261023-1017.zip'
    with zipfile.ZipFile(weights) as wz, zipfile.ZipFile(results) as rz:
        base='dico_qwen_ce_results/'; manifest=json.loads(wz.read(base+'weights_manifest.json'))
        fp=rz.read(base+'fingerprint.json')
        require(hashlib.sha256(fp).hexdigest()==manifest['fingerprint_sha256'],'已有SFT结果与权重指纹不匹配')
        require(fp==wz.read(base+'fingerprint.json'),'两个包的指纹不同')
        require(manifest['model_revision']==QREV and len(manifest['adapters'])==3,'已有老师身份不正确')
        for adapter in manifest['adapters']:
            run=adapter['run']; done=json.loads(rz.read(base+run+'/done.json'))
            require(done['best_epoch']==adapter['best_epoch'],'选中轮次不一致')
            for name,h in adapter['sha256'].items():
                b=wz.read(base+run+f"/adapters/epoch{adapter['best_epoch']}/"+name)
                require(hashlib.sha256(b).hexdigest()==h,'已有LoRA权重哈希错误')
        require({x['code_sha256'] for x in json.loads(rz.read(base+'run_meta.json'))}=={QSHA},'已有SFT代码不匹配')
    # Idempotent recovery: missing seed files/weights can be restored; never overwrite outputs.
    for archive in (results,weights):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.endswith('/'): continue
                p=dst.parent/name
                require(p.resolve().is_relative_to(dst.parent.resolve()),'越界ZIP')
                data=z.read(name)
                if p.exists():
                    # run_meta/fingerprint may be updated legitimately by the driver.
                    if '/adapters/' in name: require(sha(p)==hashlib.sha256(data).hexdigest(),'已有权重被改动')
                    continue
                p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
    q,m=modules(root)
    q.HOLDOUT_FOLD=0
    require(q.incompatible(portable_fp(read(dst/'fingerprint.json')),portable_fp(q.fingerprint()))==[], '旧Qwen结果与当前数据/配方不匹配')
    m.OUT=str(md/'dico_mdeberta_results')
    restored=0
    for task in m.stage_tasks('stage2'):
        d=Path(m.task_dir(task))
        if (d/'done.json').exists():
            require(portable_fp(read(d/'task.json'))==portable_fp(m.task_fingerprint(task,MREV)), '第五组恢复任务配置不匹配')
            restored+=1
    smoke_fp=md/'dico_mdeberta_smoke'/'fingerprint.json'
    if not smoke_fp.exists(): write(smoke_fp,{**m.common_fingerprint(),'smoke':True,'model_revision':MREV})
    split_report={}
    for k in range(5):
        qp,mp=get_parts(q,m,k)
        split_report[str(k)]=signature(qp)
    write(root/'split_audit.json',{'passed':True,'folds':split_report,'pipeline_id':pipeline_id()})
    write(root/'bundle_identity.json',{'pipeline_id':pipeline_id(),'source_bundle':read(HERE/'SHA256.json')})
    print(f'输入核验通过：第五组第二阶段已有{restored}/125项；Qwen fold0三个种子的权重可直接复用；五折师生划分逐行一致。',flush=True)

def call(root,label,args,env=None):
    log=Path(root)/'logs'/f'{label}.log'; log.parent.mkdir(exist_ok=True)
    print(f'开始 {label}；详细日志 {log}',flush=True); t0=time.time()
    with log.open('a',encoding='utf-8') as f:
        res=subprocess.run([sys.executable,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env)
    event={'task':label,'returncode':res.returncode,'minutes':(time.time()-t0)/60,'finished':time.time()}
    with (Path(root)/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
    print(f"{'完成' if res.returncode==0 else '失败'} {label}：{event['minutes']:.1f}分钟",flush=True)
    return res.returncode==0

def m_env(root,stage):
    env=os.environ.copy(); env.update(DICO_WORK=str(Path(root)/'mdeberta'),DICO_DIR=str(Path(root)/'official'),
        DICO_STAGE=stage,DICO_TEACHER=str(Path(root)/'mdeberta/dico_qwen_teacher_results'),DICO_PLATFORM='阿里云 PAI-DSW',HF_HOME=str(Path(root)/'hf_cache'),
        HF_ENDPOINT=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'),HF_HUB_DISABLE_XET='1')
    return env

def smoke(root):
    marker=Path(root)/'SMOKE_OK.json'
    if marker.exists():
        require(read(marker)['pipeline_id']==pipeline_id(),'SMOKE标记属于旧代码')
        print('当前代码SMOKE已通过；继续复用。',flush=True); budget(root); return
    jobs=[('mdeberta_smoke',[HERE/'dico_mdeberta_methods.py'],m_env(root,'smoke')),
          ('qwen_smoke',[HERE/'joint_worker.py','qwen','--root',root,'--fold',1,'--smoke'],None),
          ('kd_smoke',[HERE/'joint_worker.py','kd','--root',root,'--fold',1,'--seed',42,'--smoke'],None)]
    for label,args,env in jobs:
        if not call(root,label,args,env):
            pack(root); raise RuntimeError(f'{label}失败；正式任务未启动，请发日志')
    # Resume must skip completed Qwen/export and KD without changing output identities.
    if not call(root,'kd_smoke_resume',[HERE/'joint_worker.py','kd','--root',root,'--fold',1,'--seed',42,'--smoke']):
        raise RuntimeError('KD续跑检查失败')
    write(marker,{'passed':True,'pipeline_id':pipeline_id(),'time':time.time()})
    pack(root); budget(root); print('SMOKE全部通过。正式任务尚未开始。',flush=True)

def budget(root):
    """Pre-start estimate only; never terminate an experiment midway on a timer."""
    root=Path(root); q,m=modules(root)
    smoke_run=read(qout(root,1,True)/'track1/seed43/done.json')
    step=smoke_run['seconds_per_step']; require(isinstance(step,(float,int)) and step>0,'SMOKE缺少有效测速')
    elapsed=(time.time()-read(root/'session_clock.json')['start'])/60 if (root/'session_clock.json').exists() else 0
    per_method={'ce':1.72,'rdrop':1.93,'dir_rdrop':1.906667,'scl':1.75,'moco_m0.999':2.136667}
    md_minutes=sum(per_method[meth] for meth in METHODS for seed in range(42,47) for fold in range(5)
                   if not (root/'mdeberta/dico_mdeberta_results'/f'track1/{meth}/seed{seed}/fold{fold}/done.json').exists())
    export_smoke=read(tout(root,1,43,True)/'done.json')
    export_rate=export_smoke['minutes']/max(1,export_smoke['inputs'])
    q_minutes=teacher_minutes=kd_minutes=0
    for fold in range(5):
        parts,_=get_parts(q,m,fold)
        for seed in (42,43,44):
            if not (qout(root,fold)/'track1'/f'seed{seed}'/'done.json').exists():
                q_minutes+=max(18,step*q.steps_per_epoch(len(parts['fit']))*3/60)+2
            if not (tout(root,fold,seed)/'done.json').exists():
                teacher_minutes+=max(.5,export_rate*len(expected_inputs(q,parts)))+1
            if not (kdout(root)/f'track1/kd/seed{seed}/fold{fold}/done.json').exists():kd_minutes+=2
    remaining=md_minutes+q_minutes+teacher_minutes+kd_minutes+15
    estimate={'pipeline_id':pipeline_id(),'session_minutes_so_far':elapsed,'remaining_minutes':remaining,
              'estimated_session_calculation':(elapsed+remaining)/60*6.991,'planned_max':78,'remaining_budget_reported':94,
              'components_minutes':{'mdeberta':md_minutes,'qwen':q_minutes,'teacher':teacher_minutes,'kd':kd_minutes,'pack_and_download':15}}
    write(root/'budget_estimate.json',estimate)
    print(f"按SMOKE估算：后续约{remaining/60:.1f}小时；本次实例合计约{estimate['estimated_session_calculation']:.1f} calculation（预算78，另保留16）。",flush=True)
    require(estimate['estimated_session_calculation']<=78,'估计超过原定78 calculation，正式队列暂不启动。请发budget_estimate.json，先重新安排，不自动花掉保留额度。')

def check_scores(root,q,rs,labels,d,name,saved=None):
    """Score actual rows with official scorer; use minimal references to avoid CSV sniffer bugs."""
    from evaluation_functions.scorer import score_files
    d=Path(d); d.mkdir(parents=True,exist_ok=True)
    lookup={r['instance_id']:r for r in rows(q.data_file(1,'train','reference'))+rows(q.data_file(1,'dev','reference'))}
    ref=d/f'reference_{name}.csv'; pred=d/f'pred_{name}.csv'
    q.write_rows(str(ref),[{k:lookup[r['instance_id']][k] for k in q.REFERENCE_FIELDS} for r in rs],q.REFERENCE_FIELDS)
    q.write_rows(str(pred),[{'instance_id':r['instance_id'],'label':l} for r,l in zip(rs,labels)],['instance_id','label'])
    score=score_files(ref,pred,write_outputs=False).to_dict()
    score={k:score[k] for k in q.METRICS}
    if saved is not None:
        require(all(abs(score[k]-saved[k])<1e-8 for k in score),f'评分复算不一致 {d}/{name}')
    write(d/f'scores_{name}.json',score)
    return score

def validate(root):
    root=Path(root); q,m=modules(root); sys.path.insert(0,str(root/'official'))
    if (root/'COMPLETE.json').exists(): (root/'COMPLETE.json').unlink()
    train=q.read_rows(q.data_file(1,'train','participant_labeled')); family,fold_of=m.split_base()
    md=root/'mdeberta'/'dico_mdeberta_results'
    require(read(md/'moco_m_choice.json')['m']==0.999,'MoCo选择变化')
    counts={'mdeberta':0,'qwen':0,'teachers':0,'kd':0}; oof={}
    # Check every fifth-group task and reconstruct its OOF from logits.
    for method in METHODS:
        for seed in range(42,47):
            lms=[]
            for fold in range(5):
                d=md/f'track1/{method}/seed{seed}/fold{fold}'; done=read(d/'done.json')
                require(done['code_sha256']==MSHA and done['precision']=='bf16 autocast' and done['skipped_updates']==0,'第五组代码/精度/更新异常')
                expected=m.task_fingerprint(m.make_task(1,method,seed,fold,fold==0),MREV)
                require(portable_fp(read(d/'task.json'))==portable_fp(expected),'第五组任务指纹不一致')
                parts=m.parts_for(1,fold,fold==0,smoke=False); lm=m.load_logits(str(d/'logits_test.json')); lms.append(lm)
                for mode in ('plain','sym'):
                    labels=[m.argmax_label(m.probs(lm,r['text1'],r['text2'],mode)) for r in parts['test']]
                    check_scores(root,q,parts['test'],labels,root/'audit'/f'mdeberta/{method}/seed{seed}/fold{fold}',mode,done['scores']['test'][mode])
                counts['mdeberta']+=1
            scores={}
            for mode in ('plain','sym'):
                labels=[m.argmax_label(m.probs(lms[fold_of[family[r['pair_id']]]],r['text1'],r['text2'],mode)) for r in train]
                scores[mode]=check_scores(root,q,train,labels,root/'oof'/f'mdeberta/{method}/seed{seed}',mode)
            oof[f'mdeberta/{method}/seed{seed}']=scores
    for seed in (42,43,44):
        qlm=[]; klm=[]
        for fold in range(5):
            parts,mp=get_parts(q,m,fold); q.HOLDOUT_FOLD=fold
            source=qout(root,fold); d=source/'track1'/f'seed{seed}'; done=read(d/'done.json')
            require(read(source/'joint_driver.json')['pipeline_id']==pipeline_id(),'Qwen跨折驱动版本不一致')
            require(q.incompatible(portable_fp(read(source/'fingerprint.json')),portable_fp(q.fingerprint()))==[],'Qwen指纹不一致')
            require(read(source/'fingerprint.json')['model_revision']==QREV,'Qwen快照错误')
            require(all(x['dtype']=='torch.bfloat16' and x['code_sha256']==QSHA for x in read(source/'run_meta.json')),'Qwen来源代码/精度错误')
            require(done['best_epoch']==q.best_of(done['history']),'SFT轮次选择错误')
            qc=q.load_jsonl(str(d/'cache.jsonl')); qlm.append(qc)
            for name,rs in [('holdout',parts['test']),('dev',parts['dev'])]:
                key='fold0' if name=='holdout' and 'fold0' in done['scores'] else name
                for mode in ('plain','sym'):
                    def prob(r):
                        a=qc[q.key_of(r['text1'],r['text2'])]['p']; b=qc[q.key_of(r['text2'],r['text1'])]['p']
                        require(valid_p(a) and valid_p(b),'Qwen概率无效')
                        return a if mode=='plain' else q.sym(a,b)
                    check_scores(root,q,rs,[q.argmax_label(prob(r)) for r in rs],root/'audit'/f'qwen/fold{fold}/seed{seed}',name+'_'+mode,done['scores'][key][mode])
            counts['qwen']+=1
            _,th=validate_teacher(q,root,fold,seed,parts); counts['teachers']+=1
            kd=kdout(root)/f'track1/kd/seed{seed}/fold{fold}'; kd_done=read(kd/'done.json'); kfp=read(kd/'task.json')
            require(kd_done['code_sha256']==pipeline_id() and kd_done['precision']=='bf16 autocast' and kd_done['skipped_updates']==0,'KD代码/精度异常')
            require(kfp['teacher_sha256']==th and kfp['fold']==fold and kfp['seed']==seed and kfp['method_params']['alpha']==0.5 and kfp['method_params']['T']==2.0,'KD教师/超参数错误')
            require(kfp['joint_pipeline_id']==pipeline_id() and kfp['method_params']['source_training_sha256']==QSHA,'KD不是统一的新教师组')
            ce=read(md/f'track1/ce/seed{seed}/fold{fold}/task.json')
            for key in ('train','split','model','model_revision','data_sha256','epochs','smoke'):
                require(kfp[key]==ce[key],f'KD和CE指纹不一致:{key}')
            lm=m.load_logits(str(kd/'logits_test.json')); klm.append(lm)
            for mode in ('plain','sym'):
                labels=[m.argmax_label(m.probs(lm,r['text1'],r['text2'],mode)) for r in parts['test']]
                check_scores(root,q,parts['test'],labels,root/'audit'/f'kd/fold{fold}/seed{seed}',mode,kd_done['scores']['test'][mode])
            counts['kd']+=1
        for kind in ('qwen','kd'):
            scores={}
            for mode in ('plain','sym'):
                labels=[]
                for r in train:
                    fold=fold_of[family[r['pair_id']]]
                    if kind=='kd': p=m.probs(klm[fold],r['text1'],r['text2'],mode)
                    else:
                        a=qlm[fold][q.key_of(r['text1'],r['text2'])]['p']; b=qlm[fold][q.key_of(r['text2'],r['text1'])]['p']
                        p=a if mode=='plain' else q.sym(a,b)
                    labels.append(q.argmax_label(p))
                scores[mode]=check_scores(root,q,train,labels,root/'oof'/kind/f'seed{seed}',mode)
            oof[f'{kind}/seed{seed}']=scores
    require(counts=={'mdeberta':125,'qwen':15,'teachers':15,'kd':15},'任务数错误')
    require(len(train)==3042 and len({r['instance_id'] for r in train})==3042 and len(oof)==31,'OOF数量错误')
    report={'status':'COMPLETE','counts':counts,'oof_count':len(oof),'rows_per_oof':len(train),'pipeline_id':pipeline_id(),'oof':oof}
    report['paired_kd_minus_ce']={mode:{str(seed):{metric:oof[f'kd/seed{seed}'][mode][metric]-oof[f'mdeberta/ce/seed{seed}'][mode][metric]
                                       for metric in q.METRICS} for seed in (42,43,44)} for mode in ('plain','sym')}
    write(root/'COMPLETE.json',report)
    table=['# 英语五折验证结果','', '以下是各种子完整OOF评分的均值；dev不参与选方法。各指标、逐折记录、教师身份与原始预测均在结果包中。','',
           '| 方法 | 种子 | plain WF1 | sym WF1 | sym SoftCons | sym HardCons |','|---|---:|---:|---:|---:|---:|']
    for name,seeds in [(f'mdeberta/{x}',list(range(42,47))) for x in METHODS]+[('qwen',[42,43,44]),('kd',[42,43,44]),('mdeberta/ce',[42,43,44])]:
        means=[sum(oof[f'{name}/seed{s}'][mode][metric] for s in seeds)/len(seeds) for mode,metric in [('plain','weighted_f1'),('sym','weighted_f1'),('sym','soft_cons'),('sym','hard_cons')]]
        table.append('| '+name+' | '+str(len(seeds))+' | '+' | '.join(f'{x:.4f}' for x in means)+' |')
    table+=['','KD的配对比较使用CE种子42/43/44，不能拿五种子CE均值直接相减。新KD与历史第四组教师的KD分开报告。',
            'fold0已用于方法探索；五折结果不是从未观察过数据上的最终隐藏测试成绩。']
    (root/'结果汇总.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print('COMPLETE：125个第五组任务、15个SFT、15份教师、15个KD；31套OOF已复算。',flush=True)
    return report

def pack(root):
    root=Path(root); download=root/'download'; download.mkdir(exist_ok=True)
    stamp=time.strftime('%Y%m%d-%H%M%S'); outputs=[]
    # Stable filenames prevent repeated resume attempts from accumulating gigabytes.
    for kind in ('results','weights'):
        target=download/f'dico_joint_{kind}.zip'; tmp=target.with_suffix('.zip.tmp'); manifest={}
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED if kind=='results' else zipfile.ZIP_STORED) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file(): continue
                rel=p.relative_to(root); ps=rel.parts
                if any(x in ps for x in ('download','official','official_extract','shared','hf_cache','ms_cache','__pycache__','checkpoints','checkpoints_invalid')): continue
                if p.suffix in ('.zip','.tmp') or '.broken-' in p.name or p.name.endswith('.lock'): continue
                is_weight='adapters' in ps
                if kind=='weights' and not is_weight: continue
                if kind=='weights' and 'formal' not in ps: continue
                if kind=='results' and is_weight: continue
                z.write(p,rel.as_posix()); manifest[rel.as_posix()]=sha(p)
            z.writestr('ARCHIVE_MANIFEST.json',json.dumps({'created':stamp,'pipeline_id':pipeline_id(),'files':manifest},ensure_ascii=False))
        tmp.replace(target); outputs.append({'file':str(target),'sha256':sha(target),'bytes':target.stat().st_size})
    write(download/'DOWNLOAD.json',outputs)
    print('下载目录：',download,flush=True)

def run(root):
    require(read(Path(root)/'SMOKE_OK.json')['pipeline_id']==pipeline_id(),'先运行当前版本SMOKE')
    budget(root)
    complete=Path(root)/'COMPLETE.json'
    if complete.exists(): complete.unlink()  # Current run must revalidate; old marker is not evidence.
    failed=[]
    try:
        if not call(root,'mdeberta_stage2',[HERE/'dico_mdeberta_methods.py'],m_env(root,'stage2')): failed.append('mdeberta_stage2')
        for fold in range(5):
            for seed in (42,43,44):
                tag=f'fold{fold}_seed{seed}'
                if not call(root,'qwen_'+tag,[HERE/'joint_worker.py','qwen','--root',root,'--fold',fold,'--seed',seed]):
                    failed.append('qwen_'+tag); continue
                if not call(root,'kd_'+tag,[HERE/'joint_worker.py','kd','--root',root,'--fold',fold,'--seed',seed]): failed.append('kd_'+tag)
            pack(root)
        write(Path(root)/'queue_status.json',{'failed':failed,'pipeline_id':pipeline_id()})
        require(not failed,f'存在失败任务: {failed}')
        validate(root)
    finally: pack(root)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','smoke','run','check','pack']); p.add_argument('--root',default='/mnt/workspace/dico_joint'); a=p.parse_args()
    root=str(Path(a.root).resolve())
    if a.command=='prepare': prepare(root)
    elif a.command=='smoke': smoke(root)
    elif a.command=='run': run(root)
    elif a.command=='check':
        try: validate(root)
        finally: pack(root)
    else: pack(root)
