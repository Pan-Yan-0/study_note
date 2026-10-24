"""Audited additive recovery: preserve training identity; record export-check revision."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/mnt/workspace/a10_joint_validation')
sys.path.insert(0, str(BASE))
EXPECTED = '85740d5c93daf3af5e29e06857d70074e961c6435f7991e5ec6262fc02f639d6'
TARGETS = [(0,44),(2,44),(3,42)]

def diagnostic(q, backend, cache, new64, val_agree, val_diff, report, common):
    old32 = [x['p'] for x in backend.score([(x['a'],x['b']) for x in cache[:32]])]
    import joint_worker as worker
    ref = [x['p'] for x in cache]
    a32,d32 = worker.compare(old32,ref[:32])
    a64,d64 = worker.compare(new64,ref)
    top = lambda p: max(range(4),key=lambda i:p[i])
    changed = [{'a':x['a'],'b':x['b'],'saved':x['p'],'recomputed32':p}
               for x,p in zip(cache[:32],old32) if top(x['p'])!=top(p)]
    result = {'revision':'cache-check-original-64-v1','base_pipeline_id':EXPECTED,
              'recovery_sha256':common.sha(Path(__file__)),
              'val_agreement':val_agree,'val_max_diff':val_diff,
              'n64':len(cache),'agreement32':a32,'max_diff32':d32,
              'agreement64':a64,'max_diff64':d64,'changed32':changed}
    common.write(report,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    common.require(len(cache)==64 and val_agree>=0.98 and a64==1.0 and d64<=1e-5,
                   '同批次严格复核未通过；保留诊断，停止导出，不放宽阈值')

def patched_export(worker, common):
    source=inspect.getsource(worker.export_teacher)
    old="cache=list(q.load_jsonl(str(run/'cache.jsonl')).values())[:32]"
    new="cache=list(q.load_jsonl(str(run/'cache.jsonl')).values())[:64]"
    assert source.count(old)==1
    source=source.replace(old,new)
    old="ca,cd=compare([x['p'] for x in backend.score([(x['a'],x['b']) for x in cache])],[x['p'] for x in cache])"
    new="""new64=[x['p'] for x in backend.score([(x['a'],x['b']) for x in cache])]
    diagnostic(q,backend,cache,new64,agree,diff,d/'recovery_diagnostic.json',recovery_common)
    ca,cd=compare(new64,[x['p'] for x in cache])"""
    assert source.count(old)==1
    source=source.replace(old,new)
    namespace=dict(worker.__dict__,diagnostic=diagnostic,recovery_common=common)
    exec(compile(source,'<audited-export-check-64-v1>','exec'),namespace)
    return namespace['export_teacher']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='/mnt/workspace/dico_joint')
    p.add_argument('--worker',type=int,choices=range(3)); args=p.parse_args()
    import joint_common as c
    import joint_worker as w
    root=Path(args.root).resolve()
    c.require(c.pipeline_id()==EXPECTED,'原流水线代码已变化；停止，避免身份混用')
    c.require(c.read(root/'SMOKE_OK.json')['pipeline_id']==EXPECTED,'原SMOKE身份不符')
    # Refuse any path that could fall back to SFT retraining.
    for fold,seed in TARGETS:
        run=c.qout(root,fold)/'track1'/f'seed{seed}'
        done=c.read(run/'done.json'); trained=c.read(run/'trained.json')
        c.require(done['best_epoch']==trained['best_epoch'],'选中轮次不一致')
        c.require((run/'adapters'/f"epoch{done['best_epoch']}"/'adapter_model.safetensors').exists(),'选中权重缺失，不自动重训')
    if args.worker is not None:
        fold,seed=TARGETS[args.worker]
        w.export_teacher=patched_export(w,c)
        w.run_qwen(str(root),fold,[seed],False)
        return
    import joint_queue as queue
    record=root/'recovery'; record.mkdir(exist_ok=True)
    shutil.copy2(__file__,record/'recover_joint.py')
    c.write(record/'provenance.json',{'base_pipeline_id':EXPECTED,'recovery_sha256':c.sha(Path(__file__)),
          'targets':TARGETS,'change':'Only export validation replay uses original first 64 cache rows; training and teacher scoring unchanged.',
          'strict_check':'cache argmax agreement == 1 and max probability difference <= 1e-5',
          'status':'running'})
    failures=[]
    try:
        for i,(fold,seed) in enumerate(TARGETS):
            tag=f'fold{fold}_seed{seed}'
            if not queue.call(str(root),'recovery_qwen_'+tag,[Path(__file__),'--root',root,'--worker',i]):
                failures.append('qwen_'+tag); continue
            if not queue.call(str(root),'recovery_kd_'+tag,[BASE/'joint_worker.py','kd','--root',root,'--fold',fold,'--seed',seed]):
                failures.append('kd_'+tag)
        c.write(record/'status.json',{'failed':failures})
        c.require(not failures,f'补跑未全部通过: {failures}；请发recovery日志')
        queue.validate(str(root))
        # Preserve the historical queue_status; COMPLETE is the final acceptance marker.
        c.write(record/'status.json',{'failed':[],'status':'COMPLETE'})
    finally:
        queue.pack(str(root))

if __name__=='__main__': main()
