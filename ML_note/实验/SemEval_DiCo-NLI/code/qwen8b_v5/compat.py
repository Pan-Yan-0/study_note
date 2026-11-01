"""4B compatibility checks; independent subprocesses for deterministic short runs."""
import argparse, os, shutil
from common import *
R4='cdbee75f17c01a7cc42f958dc650907174af0554'
def main():
    p=argparse.ArgumentParser(); p.add_argument('kind',choices=['download','zero','old','new','compare']); p.add_argument('--root',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True); p.add_argument('--baseline-cache',type=Path,required=True); a=p.parse_args()
    a.root=a.root.resolve(); a.data=a.data.resolve(); a.adapter=a.adapter.resolve(); a.baseline_cache=a.baseline_cache.resolve()
    require(a.adapter.is_dir(),'需要已有4B LoRA目录')
    source={'pipeline':pipeline(),'adapter':{x.name:sha(x) for x in a.adapter.iterdir() if x.is_file()},'baseline_cache':sha(a.baseline_cache),'data':read(HERE/'expected_data.json')}
    bind(a.root/'compat_identity.json',source)
    if a.kind in ('old','new'):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from worker import load,runtime
    q=load(a.root,a.data,0,True,8)
    # old: direct upstream defaults. new: explicit adapter overrides, same 4B recipe.
    if a.kind=='old':
        q.MODEL='Qwen/Qwen3-4B-Instruct-2507'; q.MODEL_REVISION=R4
    else:
        for k,v in {'MODEL':'Qwen/Qwen3-4B-Instruct-2507','MODEL_REVISION':R4,'TRAIN_BATCH':8,'GRAD_ACC':2}.items(): setattr(q,k,v)
    q.MAX_EPOCHS=3
    if a.kind=='download':
        path,rev,src=q.fetch_from_modelscope(R4); require(rev==R4,'4B版本不同'); base=Path(path)
        write(a.root/'model4b.json',{'path':path,'revision':rev,'source':src,'files':{str(f.relative_to(base)):sha(f) for f in base.rglob('*') if f.is_file()}}); return
    env=runtime(); bind(a.root/'runtime.json',env)
    if a.kind=='compare':
        import torch
        from safetensors.torch import load_file
        old=a.root/'compat-old'; new=a.root/'compat-new'
        require(read(old/'inputs.json')==read(new/'inputs.json'),'输入或目标不一致')
        x=read(old/'first_batch.json'); y=read(new/'first_batch.json')
        require(all(x[k]==y[k] for k in ('input_ids','attention_mask','answer_ids','loss_positions')),'首批输入/损失位置不同')
        require(abs(x['loss']-y['loss'])<=1e-5,'首批损失不同')
        require(max(abs(a-b) for px,py in zip(x['probabilities'],y['probabilities']) for a,b in zip(px,py))<=1e-5,'首批概率不同')
        for ep in (1,2,3):
            x=load_file(str(old/f'update{ep}.safetensors')); y=load_file(str(new/f'update{ep}.safetensors'))
            require(set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in x),'参数更新不一致')
        require(read(old/'info.json')['steps']==read(new/'info.json')['steps']==3,'不是3次更新')
        require(read(a.root/'zero4b.json')['passed'],'零样本不通过')
        updates={str(f.relative_to(a.root)):sha(f) for d in (old,new) for f in d.glob('update*.safetensors')}
        write(a.root/'parameter_comparison.json',{'passed':True,'updates':updates,'tolerance':0,'steps':3})
        for d in (old,new):
            for f in d.glob('update*.safetensors'): f.unlink()
        artifacts=[a.root/'zero4b.json',a.root/'compat_identity.json',a.root/'parameter_comparison.json']+[d/n for d in (old,new) for n in ('inputs.json','info.json','first_batch.json')]
        write(a.root/'COMPATIBLE.json',{'identity':source,'runtime':env,'passed':True,'files':{str(f.relative_to(a.root)):sha(f) for f in artifacts}}); return
    rec=read(a.root/'model4b.json')
    for n,h in rec['files'].items(): require(sha(Path(rec['path'])/n)==h,'4B缓存变化')
    q.fetch_model=lambda revision:(rec['path'],R4,rec['source'])
    b=q.QwenSFT(R4)
    if a.kind=='zero':
        rows=q.track_parts(1,q.make_split(),False)['dev']; inputs=q.needed_inputs(rows); baseline=q.load_jsonl(str(a.baseline_cache))
        require(all(q.key_of(*x) in baseline for x in inputs),'历史缓存缺少dev输入')
        b.load_base(); largest=0
        for batch in q.chunks(inputs,64):
            values=b.score(batch); require(len(values)==len(batch),'零样本缺行')
            for ab,value in zip(batch,values):
                old=baseline[q.key_of(*ab)]['p']; new=value['p']; require(valid_p(old) and valid_p(new),'概率非法')
                diff=max(abs(x-y) for x,y in zip(old,new)); largest=max(largest,diff)
                require(diff<=1e-5 and old.index(max(old))==new.index(max(new)),'历史零样本差异超限')
        write(a.root/'zero4b.json',{'passed':True,'inputs':len(inputs),'max_diff':largest,'identity':source}); b.release(); return
    import torch,peft
    from safetensors.torch import load_file
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.use_deterministic_algorithms(True)
    saved=load_file(str(a.adapter/'adapter_model.safetensors')); original=peft.get_peft_model
    def wrap(*args,**kwargs):
        model=original(*args,**kwargs); current=peft.get_peft_model_state_dict(model)
        require(set(current)==set(saved) and all(current[k].shape==saved[k].shape for k in current),'已有adapter不匹配冻结LoRA')
        peft.set_peft_model_state_dict(model,saved)
        require(all(torch.equal(v.detach().cpu().to(saved[k].dtype),saved[k]) for k,v in peft.get_peft_model_state_dict(model).items()),'初始adapter重载不一致')
        return model
    peft.get_peft_model=wrap
    d=a.root/('compat-'+a.kind); require(not d.exists(),'兼容性目录已存在；请保留原目录，使用新root重新验收'); d.mkdir(parents=True)
    parts=q.track_parts(1,q.make_split(),False); fit=q.smoke_pairs(parts['fit'],16); require(len(fit)==16,'短程检查必须恰好16行'); val=q.smoke_pairs(parts['val'],32)
    write(d/'inputs.json',{'tokens':[b.encode(r['text1'],r['text2']) for r in fit],'labels':[r['label'] for r in fit],'target_ids':b.target_ids})
    first={}
    original_logits=b._last_logits
    def capture_logits(model,ids,mask):
        logits=original_logits(model,ids,mask)
        if model.training and not first:
            first.update(input_ids=ids.detach().cpu().tolist(),attention_mask=mask.detach().cpu().tolist(),loss_positions=(mask.sum(-1)-1).detach().cpu().tolist(),probabilities=b.letter_logprobs(logits).exp().detach().float().cpu().tolist())
        return logits
    b._last_logits=capture_logits
    import torch.nn.functional as F
    original_ce=F.cross_entropy
    def capture_ce(logits,target,*args,**kwargs):
        result=original_ce(logits,target,*args,**kwargs)
        if first and 'loss' not in first and b.model.training:
            first.update(answer_ids=target.detach().cpu().tolist(),loss=float(result.detach()))
            write(d/'first_batch.json',first)
        return result
    F.cross_entropy=capture_ce
    original_write=q.write_json
    def capture(path,obj):
        original_write(path,obj)
        if Path(path).name=='history.json':
            ep=obj[-1]['epoch']; shutil.copyfile(d/'adapters'/f'epoch{ep}'/'adapter_model.safetensors',d/f'update{ep}.safetensors')
    q.write_json=capture
    try:
        with sdpa_kernel(SDPBackend.MATH): history,info=b.train(str(d),fit,val,42)
        write(d/'info.json',info)
    finally: b.release()
if __name__=='__main__': main()
