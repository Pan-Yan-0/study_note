"""One isolated Qwen+export or KD task, using unchanged reviewed training kernels."""
import argparse
import os
import time
from pathlib import Path
from joint_common import *

def score_qwen(q, backend, d, parts):
    cache_path=str(d/'cache.jsonl'); cache=q.load_jsonl(cache_path)
    inputs=q.needed_inputs(parts['test']+parts['dev']); t0=time.time()
    todo=[x for x in inputs if q.key_of(*x) not in cache]
    for batch in q.chunks(sorted(todo,key=lambda x:-len(x[0])-len(x[1])),64):
        got=backend.score(batch)
        items=[{'a':a,'b':b,**v} for (a,b),v in zip(batch,got)]
        require(len(got)==len(batch) and all(valid_p(x['p']) for x in items),'无效Qwen输出')
        q.append_jsonl(cache_path,items); cache.update({q.key_of(x['a'],x['b']):x for x in items})
    scores={}
    for name,split,rs in [('holdout','train',parts['test']),('dev','dev',parts['dev'])]:
        ref=q.reference_for(1,split,rs,str(d),name); scores[name]={}
        for mode in ('plain','sym'):
            def prob(r):
                a=cache[q.key_of(r['text1'],r['text2'])]['p']
                b=cache[q.key_of(r['text2'],r['text1'])]['p']
                return a if mode=='plain' else q.sym(a,b)
            pred=str(d/f'pred_{name}_{mode}.csv')
            q.write_rows(pred,[{'instance_id':r['instance_id'],'label':q.argmax_label(prob(r))} for r in rs],['instance_id','label'])
            scores[name][mode]=q.official_score(ref,pred,str(d/f'scores_{name}_{mode}'))
    return scores, {'inputs':len(inputs),'scored_this_session':len(todo),'score_seconds_this_session':time.time()-t0}

def compare(a,b):
    require(len(a)==len(b) and len(a)>0,'预测长度不匹配')
    require(all(valid_p(p) for p in a+b),'无效对照概率')
    top=lambda p:max(range(4),key=lambda i:p[i])
    return sum(top(p)==top(r) for p,r in zip(a,b))/len(a),max(abs(x-y) for p,r in zip(a,b) for x,y in zip(p,r))

def export_teacher(q, backend, root, fold, seed, parts, smoke):
    d=tout(root,fold,seed,smoke); run=Path(q.OUT)/'track1'/f'seed{seed}'
    if (d/'done.json').exists():
        validate_teacher(q,root,fold,seed,parts,smoke)
        print('教师已导出且复核通过，跳过',fold,seed,flush=True); return
    sd=read(run/'done.json'); best=sd['best_epoch']
    adapter=run/'adapters'/f'epoch{best}'/'adapter_model.safetensors'
    require(adapter.exists(),'选中轮次权重缺失；不静默重训已完成任务')
    meta={'version':'dico_joint_teacher-1','pipeline_id':pipeline_id(),'fold':fold,'seed':seed,
          'smoke':smoke,'labels':LABELS,'split':signature(parts),'best_epoch':best,
          'adapter_sha256':sha(adapter),'source_fingerprint_sha256':sha(Path(q.OUT)/'fingerprint.json'),
          'source_training_sha256':QSHA,'source_done_sha256':sha(run/'done.json')}
    if (d/'identity.json').exists(): require(read(d/'identity.json')==meta,'旧教师缓存来源不同')
    write(d/'identity.json',meta)
    # train() with trained.json only reloads and verifies; no optimizer update.
    if backend.model is None: backend.train(str(run),parts['fit'],parts['val'],seed)
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    saved=load_file(str(adapter)); now=get_peft_model_state_dict(backend.model)
    require(set(saved)==set(now) and all(torch.equal(now[k].detach().cpu().to(v.dtype),v) for k,v in saved.items()),'当前模型不是选中的教师权重')
    backend.model.eval(); t0=time.time()
    vi=list(dict.fromkeys((r['text1'],r['text2']) for r in parts['val']))
    new=[x['p'] for x in backend.score(vi)]
    agree,diff=compare(new,read(run/'val_probs'/f'epoch{best}.json'))
    cache=list(q.load_jsonl(str(run/'cache.jsonl')).values())[:32]
    ca,cd=compare([x['p'] for x in backend.score([(x['a'],x['b']) for x in cache])],[x['p'] for x in cache])
    checks={'val_argmax_agreement':agree,'val_max_prob_diff':diff,'cache_argmax_agreement':ca,
            'cache_max_prob_diff':cd,'tensors_equal':len(saved)}
    require(agree>=0.98 and ca>=0.98,'教师重新载入后的预测与SFT不一致')
    path=d/'teacher.jsonl'
    # Repair only an interrupted final line, then independently reject duplicate/out-of-split records.
    q.load_jsonl(str(path))
    have=teacher_rows(q,path,parts,False) if path.exists() else {}
    want=expected_inputs(q,parts)
    for part in ('fit','val'):
        todo=[x for x in q.needed_inputs(parts[part]) if q.key_of(*x) not in have]
        for batch in q.chunks(sorted(todo,key=lambda x:-len(x[0])-len(x[1])),64):
            got=backend.score(batch); require(len(got)==len(batch),'教师推理缺行')
            items=[{'a':a,'b':b,'part':part,**v} for (a,b),v in zip(batch,got)]
            require(all(valid_p(x['p']) for x in items),'无效教师概率')
            q.append_jsonl(str(path),items)
    by=teacher_rows(q,path,parts)
    write(d/'done.json',{'identity_sha256':sha(d/'identity.json'),'teacher_sha256':sha(path),
                        'checks':checks,'inputs':len(want),'fit_rows':len(parts['fit']),
                        'minutes':(time.time()-t0)/60})
    validate_teacher(q,root,fold,seed,parts,smoke)
    print(f'教师导出完成 fold{fold} seed{seed}: {len(by)}个输入，val一致率{agree:.3f}',flush=True)

def run_qwen(root,fold,seeds,smoke):
    q,m=modules(root,smoke); q.HOLDOUT_FOLD=fold; q.SEEDS=list(seeds)
    q.OUT=str(qout(root,fold,smoke)); Path(q.OUT).mkdir(parents=True,exist_ok=True)
    parts,_=get_parts(q,m,fold,smoke)
    q.ensure_packages(); q.check_fingerprint(q.fingerprint())
    driver=Path(q.OUT)/'joint_driver.json'
    ident={'pipeline_id':pipeline_id(),'fold':fold,'smoke':smoke,'split':signature(parts),
           'training_source_sha256':QSHA,'legacy_fold0_score_key':'fold0 (only imported original results)'}
    if driver.exists(): require(read(driver)==ident,'Qwen续跑代码/划分变化')
    write(driver,ident); cache_fetch(q,root,'qwen',QREV)
    backend=q.QwenSFT(QREV); require(backend.bf16,'本队列只支持A10 bf16')
    q.record_revision(backend.revision); q.record_run({**backend.info(),'joint_driver':ident})
    if smoke: q.self_check_losses()
    try:
        for seed in seeds:
            d=Path(q.OUT)/'track1'/f'seed{seed}'; d.mkdir(parents=True,exist_ok=True)
            if not (d/'done.json').exists():
                t0=time.time(); interrupt=None
                if smoke and seed==seeds[0] and not (d/'checkpoints').exists() and not (d/'trained.json').exists():
                    interrupt=q.steps_per_epoch(len(parts['fit']))+2
                interrupted=False
                try: history,info=backend.train(str(d),parts['fit'],parts['val'],seed,interrupt)
                except q.SimulatedInterrupt: interrupted=True
                if interrupted:
                    q.plant_broken_checkpoint(str(d)); backend.release()
                    history,info=backend.train(str(d),parts['fit'],parts['val'],seed)
                scores,stats=score_qwen(q,backend,d,parts)
                write(d/'done.json',{'scores':scores,'stats':stats,'history':history,**info,
                                    'fold':fold,'seed':seed,'pipeline_id':pipeline_id(),
                                    'minutes_last_session':(time.time()-t0)/60})
                print(f'SFT完成 fold{fold} seed{seed}',flush=True)
            export_teacher(q,backend,root,fold,seed,parts,smoke)
            backend.release()
        if smoke:
            first=read(Path(q.OUT)/'track1'/f'seed{seeds[0]}'/'done.json')
            require(first['resumed_from_epoch']==1,'Qwen SMOKE没有从检查点恢复')
            require(any(x['checkpoint']=='checkpoint-999999' for x in first['resume']['skipped']),'未拒绝损坏检查点')
            require(all(c['val_argmax_agreement']>=0.98 for c in first['adapter_checks']),'LoRA重载检查失败')
            write(Path(q.OUT)/'smoke_checks.json',{'passed':True,'pipeline_id':pipeline_id(),
                    'resume':first['resume'],'seconds_per_step':first['seconds_per_step']})
    finally: backend.release()

def run_kd(root,fold,seed,smoke):
    q,m=modules(root,smoke); qp,mp=get_parts(q,m,fold,smoke)
    tp,tsha=validate_teacher(q,root,fold,seed,qp,smoke)
    m.OUT=str(kdout(root,smoke)); m.VERSION='dico_joint_kd-1'
    if smoke: m.SMOKE_ROWS=dict(q.SMOKE_ROWS)
    original_common=m.common_fingerprint
    m.common_fingerprint=lambda:{**original_common(),'joint_pipeline_id':pipeline_id(),
                                  'teacher_definition':'Qwen methods v2 CE, matched fold and seed'}
    old_params=m.method_params
    m.method_params=lambda meth: ({'loss':'0.5*CE + 0.5*T^2*KL(teacher_T || student_T)',**m.KD,
                                  'teacher':'Qwen methods v2 CE, matched fold and seed','source_training_sha256':QSHA}
                                 if meth=='kd' else old_params(meth))
    m.code_sha256=lambda:(pipeline_id(),'joint pipeline sources SHA256')
    original_parts=m.parts_for
    m.parts_for=lambda track,k,dev,**kw: mp if track==1 and k==fold else original_parts(track,k,dev,**kw)
    def teacher(s,fit):
        require(s==seed and signature({'fit':fit,'val':mp['val'],'test':mp['test']})==signature(mp),'KD请求了错误教师')
        return tp,tsha
    m.load_teacher=teacher
    m.ensure_packages(); m.check_common_fingerprint()
    cache_fetch(m,root,'mdeberta',MREV)
    path,rev,source=m.fetch_model(MREV); m._MODEL_PATH['path']=path
    import torch
    require(torch.cuda.is_available() and m.amp_dtype()==torch.bfloat16,'KD需要A10 bf16')
    m.ENV.update({**m.env_info(),'platform':m.PLATFORM,'amp':'bf16'})
    # Same student settings/data as CE; environment software differences are not silently hidden.
    if not smoke:
        ce=Path(root)/'mdeberta'/'dico_mdeberta_results'/f'track1/ce/seed{seed}/fold{fold}'
        cf=read(ce/'task.json'); cd=read(ce/'done.json')
        candidate=m.task_fingerprint(m.make_task(1,'kd',seed,fold,fold==0),MREV)
        for key in ('model','train','split','model_revision','track','fold','seed','data_sha256','epochs','smoke'):
            require(cf[key]==candidate[key],f'KD和CE不匹配: {key}')
        require(cd['precision']=='bf16 autocast' and cd['skipped_updates']==0,'CE精度或更新记录异常')
        require(m.ENV['gpu']==cd['env']['gpu'] and m.ENV['torch']==cd['env']['torch'],'CE与KD GPU/torch不匹配')
    h=m.record_run({**m.ENV,'model_revision':rev,'model_source':source})
    t=m.make_task(1,'kd',seed,fold,smoke or fold==0)
    # Preserve exactly the original with_dev policy outside SMOKE.
    if not smoke and fold!=0: mp.pop('dev',None)
    m.run_task(t,rev,h)
    done=read(Path(m.task_dir(t))/'done.json')
    require(done['precision']=='bf16 autocast' and done['skipped_updates']==0,'KD回退精度或跳过更新')
    m.summarize()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('kind',choices=['qwen','kd']); p.add_argument('--root',required=True)
    p.add_argument('--fold',type=int,required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    require(a.fold in range(5) and a.seed in (42,43,44),'非法fold/seed')
    if a.kind=='qwen': run_qwen(a.root,a.fold,[42,43] if a.smoke else [a.seed],a.smoke)
    else: run_kd(a.root,a.fold,a.seed,a.smoke)
