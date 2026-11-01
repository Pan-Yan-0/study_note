"""One GPU job per subprocess. Simulated interruption exits before reload."""
import argparse, importlib.util, importlib.metadata, os, sys, platform, time
from common import *
def load(root,data,fold,smoke,batch):
    os.environ.update(DICO_WORK=str(root),DICO_DIR=str(data),DICO_METHOD='ce',DICO_SMOKE='1' if smoke else '0',HF_ENDPOINT=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'))
    spec=importlib.util.spec_from_file_location('v5_upstream',HERE/'upstream_qwen.py'); q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    q.MODEL=MODEL; q.MODEL_REVISION=REV; q.HOLDOUT_FOLD=fold
    q.TRAIN_BATCH=batch; q.GRAD_ACC=16//batch; q.LORA_DROPOUT=0
    require((data/'final_data').is_dir(),'请先解压官方数据并指定 --data')
    require(q.fingerprint()['data_sha256']==read(HERE/'expected_data.json'),'官方数据与原验收版本不一致')
    return q

def runtime():
    import torch
    require(torch.cuda.is_available() and 'A10' in torch.cuda.get_device_name(0),'此冻结队列需要 A10')
    require(torch.cuda.is_bf16_supported(),'需要原生 bf16')
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'precision':'bf16','packages':{k:importlib.metadata.version(k) for k in ('transformers','peft','accelerate','datasets','numpy','safetensors','tokenizers','huggingface_hub')}}

def run(a):
    root=Path(a.root).resolve(); data=Path(a.data).resolve(); root.mkdir(parents=True,exist_ok=True)
    q=load(root,data,a.fold,a.kind in ('interrupt','resume','smoke','probe'),a.batch)
    ident={'pipeline':pipeline(),'model':MODEL,'revision':REV,'data':q.fingerprint()['data_sha256'],'batch':a.batch,'grad_acc':16//a.batch}
    bind(root/'identity.json',ident)
    if a.kind=='download':
        path,rev,source=q.fetch_from_modelscope(REV)
        require(rev==REV,'模型快照不同')
        p=Path(path); files={str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
        write(root/'model.json',{'path':str(p),'revision':rev,'source':source,'files':files}); return
    record=read(root/'model.json'); require(record['revision']==REV,'模型版本错误')
    for n,h in record['files'].items(): require(sha(Path(record['path'])/n)==h,'模型缓存内容变化: '+n)
    q.fetch_model=lambda revision:(record['path'],REV,record['source'])
    env=runtime(); bind(root/'runtime.json',env)
    for k,v in q.PINS.items(): require(env['packages'][k]==v,'请先安装固定依赖: '+k)
    smoke=a.kind!='formal'; name='probe' if a.kind=='probe' else 'smoke' if smoke else 'formal'
    d=root/name/f'seed{a.seed}'/f'fold{a.fold}'; d.mkdir(parents=True,exist_ok=True)
    q.OUT=str(d); q.SEEDS=[a.seed]
    full=q.track_parts(1,q.make_split(),False); parts=q.track_parts(1,q.make_split(),smoke)
    if smoke:
        # Keep full validation and include longest fit inputs, while preserving pairs.
        parts['val']=full['val']
    task={**ident,'runtime':env,'fold':a.fold,'seed':a.seed,'smoke':smoke,'settings':q.fingerprint(),'split_ids':{k:[r['instance_id'] for r in v] for k,v in parts.items()}}
    bind(d/'task.json',task)
    if (d/'COMPLETE.json').exists(): verify(d,task); print('已完成并校验，跳过',d); return
    if (d/'trained.json').exists():
        best=read(d/'trained.json')['best_epoch']; require((d/'adapters'/f'epoch{best}'/'adapter_model.safetensors').exists(),'缺少已选中权重；请恢复权重包，不自动重训')
    b=q.QwenSFT(REV); require(b.bf16,'精度回退')
    original_score=b.score
    def checked_score(inputs):
        got=original_score(inputs); require(len(got)==len(inputs) and all(valid_p(x['p']) for x in got),'模型输出缺行或概率非法'); return got
    b.score=checked_score
    for x in q.load_jsonl(str(d/'cache.jsonl')).values(): require(valid_p(x['p']),'已有概率缓存损坏')
    rendered=b.chat(q.build_prompt('a dog','an animal'))+q.MARKER
    require(rendered.endswith('Answer:') and '<think>' in rendered and '</think>' in rendered and rendered.split('<think>',1)[1].split('</think>',1)[0].strip()=='','8B空think模板异常')
    if smoke:
        longest=sorted(full['fit'],key=lambda r:len(b.encode(r['text1'],r['text2'])),reverse=True)[:a.batch]
        pair_ids={r['pair_id'] for r in longest}; extra=[r for r in full['fit'] if r['pair_id'] in pair_ids]
        by={r['instance_id']:r for r in parts['fit']+extra}; parts['fit']=list(by.values())
        if a.kind=='probe':
            parts['fit']=sorted(full['fit'],key=lambda r:len(b.encode(r['text1'],r['text2'])),reverse=True)[:16]
        # Actual training split is recorded independently before any updates.
    bind(d/'actual_split.json',{k:[r['instance_id'] for r in v] for k,v in parts.items()})
    write(d/'prompt.json',{'rendered':b.chat(q.build_prompt('a dog','an animal'))+q.MARKER,'targets':b.target_ids,'max_fit_tokens':max(len(b.encode(r['text1'],r['text2'])) for r in full['fit'])})
    if a.kind=='resume':
        require((d/'INTERRUPTED.json').exists(),'缺少真实中断记录')
        q.plant_broken_checkpoint(str(d))
    start=time.time()
    try:
        if smoke: q.self_check_losses()
        try:
            history,info=b.train(str(d),parts['fit'],parts['val'],a.seed,q.steps_per_epoch(len(parts['fit']))+2 if a.kind=='interrupt' else None)
        except q.SimulatedInterrupt:
            write(d/'INTERRUPTED.json',{'pipeline':pipeline(),'kind':'intentional','resume_in_new_process':True})
            return 75
        require(a.kind!='interrupt','模拟中断未触发')
        scores,stats=q.score_and_evaluate(b,str(d),1,parts)
        # Upstream labels this key fold0 for every fold; normalize in v5 output.
        scores['holdout']=scores.pop('fold0')
        write(d/'done.json',{'scores':scores,'stats':stats,'history':history,**info,'fold':a.fold,'seed':a.seed,'seconds':time.time()-start})
        if a.kind=='resume':
            require(info['resumed_from_epoch']==1,'未从第一轮恢复')
            require(any(x['checkpoint']=='checkpoint-999999' for x in info['resume']['skipped']),'坏检查点未拒绝')
        seal(d,task)
    finally: b.release()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('kind',choices=['download','interrupt','resume','smoke','probe','formal']); p.add_argument('--root',required=True); p.add_argument('--data',required=True); p.add_argument('--fold',type=int,choices=range(5),default=0); p.add_argument('--seed',type=int,choices=[42,43,44],default=42); p.add_argument('--batch',type=int,choices=[4,8],default=8)
    a=p.parse_args()
    try: sys.exit(run(a) or 0)
    except Exception:
        import traceback
        write(Path(a.root)/'last_error.json',{'kind':a.kind,'fold':a.fold,'seed':a.seed,'traceback':traceback.format_exc()}); raise
