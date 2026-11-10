"""4B compatibility checks (unchanged from qwen8b_v5.5 except comments); independent subprocesses for deterministic short runs.
零样本核对：模型、模板和打分与历史 4B 缓存一致；old / new：本队列的覆盖入口和上游默认入口 3 次更新逐位相同。"""
import argparse, os, shutil
from common import *
R4='cdbee75f17c01a7cc42f958dc650907174af0554'
def sixteen(q,rows):
    """恰好 16 行、整对、四类都有：批 8 × 累积 2 时每轮正好 1 次更新，3 轮 3 次。
    fold 0 上 smoke_pairs(fit, 16) 会给出 17 行（只能整对加），所以先取 15 行，再按固定顺序补一条单条的 O。"""
    import random
    base=q.smoke_pairs(rows,15); keep={r['pair_id'] for r in base}
    if len(base)==15:
        count={}
        for r in rows: count[r['pair_id']]=count.get(r['pair_id'],0)+1
        singles=sorted(p for p,n in count.items() if n==1 and p not in keep)
        random.Random(q.FOLD_SEED).shuffle(singles); keep.add(singles[0])
    fit=[r for r in rows if r['pair_id'] in keep]
    require(len(fit)==16 and {r['label'] for r in fit}==set(q.LABELS),'短程检查必须恰好16行且四类都有')
    return fit
def main():
    p=argparse.ArgumentParser(); p.add_argument('kind',choices=['download','zero','old','new','compare']); p.add_argument('--root',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True); p.add_argument('--baseline-cache',type=Path,required=True); a=p.parse_args()
    a.root=a.root.resolve(); a.data=a.data.resolve(); a.adapter=a.adapter.resolve(); a.baseline_cache=a.baseline_cache.resolve()
    require(a.adapter.is_dir(),'需要已有4B LoRA目录')
    source={'pipeline':pipeline(),'adapter':{x.name:sha(x) for x in a.adapter.iterdir() if x.is_file()},'baseline_cache':sha(a.baseline_cache),'data':read(HERE/'expected_data.json')}
    bind(a.root/'compat_identity.json',source)
    if a.kind in ('old','new'):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from worker import load,runtime,fetch_verified
    # old：上游原样（联合验证 4B 用的就是这些默认值），不改任何全局设置，SMOKE=0 所以 MAX_EPOCHS 本来就是 3。
    # new：走本队列 worker 的覆盖路径（再显式写一遍 4B 的值），检验覆盖机制和上游默认等价。
    q=load(a.root,a.data,0,False,8,overrides=(a.kind!='old'))
    if a.kind!='old':
        for k,v in {'MODEL':'Qwen/Qwen3-4B-Instruct-2507','MODEL_REVISION':R4,'TRAIN_BATCH':8,'GRAD_ACC':2,'MAX_EPOCHS':3,'LORA_DROPOUT':0.0,'HOLDOUT_FOLD':0}.items(): setattr(q,k,v)
    require(q.MODEL=='Qwen/Qwen3-4B-Instruct-2507' and q.MODEL_REVISION==R4 and q.MAX_EPOCHS==3,'兼容性检查必须是 4B、3 轮')
    if a.kind=='download':
        path,rev,src=fetch_verified(q,R4); require(rev==R4,'4B版本不同'); base=Path(path)
        write(a.root/'model4b.json',{'path':path,'revision':rev,'source':src,'files':{str(f.relative_to(base)):sha(f) for f in base.rglob('*') if f.is_file()}}); return
    env=runtime(); bind(a.root/'runtime.json',env)
    if a.kind=='compare':
        import torch
        from safetensors.torch import load_file
        old=a.root/'compat-old'; new=a.root/'compat-new'
        require(read(old/'inputs.json')==read(new/'inputs.json'),'输入或目标不一致')
        require(read(old/'settings.json')==read(new/'settings.json'),'v5 覆盖后的训练设置和上游默认不同')
        x=read(old/'first_batch.json'); y=read(new/'first_batch.json')
        require(x['context']==y['context'] and x['context']['training'] and x['context']['grad_enabled'],'首批不是同一种真实训练前向')
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
        artifacts=[a.root/'zero4b.json',a.root/'compat_identity.json',a.root/'parameter_comparison.json']+[d/n for d in (old,new) for n in ('inputs.json','settings.json','info.json','first_batch.json')]
        write(a.root/'COMPATIBLE.json',{'identity':source,'runtime':env,'passed':True,'files':{str(f.relative_to(a.root)):sha(f) for f in artifacts}}); return
    rec=read(a.root/'model4b.json')
    for n,h in rec['files'].items(): require(sha(Path(rec['path'])/n)==h,'4B缓存变化')
    q.fetch_model=lambda revision:(rec['path'],R4,rec['source'])
    b=q.QwenSFT(R4)
    if a.kind=='zero':
        # 完全按历史缓存产生时的方式重算（10-22 dico_qwen_sft-4 的 score_and_evaluate）：fold 0 留出 + dev 的全部有序输入，
        # 按两段文本长度之和降序排序，每 64 个一批。bf16 下批次组成不同会改变数值（11-01 A10 实测：只算 dev、原顺序分批时
        # 最大差 0.354、1 个最大类不同；按历史方式为 0），所以这里复现批次，门槛不放宽。
        parts=q.track_parts(1,q.make_split(),False); baseline=q.load_jsonl(str(a.baseline_cache))
        inputs=q.needed_inputs(parts['test']+parts['dev'])
        require(len(inputs)==len(baseline)==1448 and all(q.key_of(*x) in baseline for x in inputs),'历史缓存和 fold0+dev 的输入集合对不上')
        b.load_base()
        def run(batches):
            largest=0; flips=0; n=0
            for batch in batches:
                values=b.score(batch); require(len(values)==len(batch),'零样本缺行')
                for ab,value in zip(batch,values):
                    old=baseline[q.key_of(*ab)]['p']; new=value['p']; require(valid_p(old) and valid_p(new),'概率非法')
                    largest=max(largest,max(abs(x-y) for x,y in zip(old,new))); flips+=old.index(max(old))!=new.index(max(new)); n+=1
            return {'inputs':n,'max_diff':largest,'argmax_flips':flips}
        gate=run(q.chunks(sorted(inputs,key=lambda x:-len(x[0])-len(x[1])),64))
        # 诊断（不作门槛）：只算 dev、按原顺序分批，记录批次组成带来的数值差
        diag=run(q.chunks(q.needed_inputs(parts['dev']),64))
        passed=gate['max_diff']<=1e-5 and gate['argmax_flips']==0
        write(a.root/'zero4b.json',{'passed':passed,'procedure':'fold0+dev，按长度降序，每批64（复现历史缓存）','tolerance':1e-5,**gate,
                                    'diagnostic_dev_natural_order':diag,'identity':source})
        b.release(); require(passed,'历史零样本差异超限：%s'%gate); return
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
    parts=q.track_parts(1,q.make_split(),False); fit=sixteen(q,parts['fit']); val=q.smoke_pairs(parts['val'],32)
    d=a.root/('compat-'+a.kind); require(not d.exists(),'兼容性目录已存在；请保留原目录，使用新root重新验收'); d.mkdir(parents=True)
    write(d/'inputs.json',{'tokens':[b.encode(r['text1'],r['text2']) for r in fit],'labels':[r['label'] for r in fit],'target_ids':b.target_ids})
    write(d/'settings.json',q.fingerprint())  # 含模型、LoRA、批大小、轮数、学习率、划分和数据哈希
    first={}
    original_logits=b._last_logits
    calls=[0]
    def capture_logits(model,ids,mask):
        logits=original_logits(model,ids,mask)
        calls[0]+=1
        if model.training and torch.is_grad_enabled() and not first:
            # 上游 ForwardCheck 在 model.eval()、no_grad 下调用，不会被记下；这里记的是第一次真实训练前向
            first.update(context={'training':True,'grad_enabled':True,'call_index':calls[0]},input_ids=ids.detach().cpu().tolist(),attention_mask=mask.detach().cpu().tolist(),loss_positions=(mask.sum(-1)-1).detach().cpu().tolist(),probabilities=b.letter_logprobs(logits).exp().detach().float().cpu().tolist())
        return logits
    b._last_logits=capture_logits
    import torch.nn.functional as F
    original_ce=F.cross_entropy
    def capture_ce(logits,target,*args,**kwargs):
        result=original_ce(logits,target,*args,**kwargs)
        if first and 'loss' not in first and b.model.training and torch.is_grad_enabled() and calls[0]==first['context']['call_index']:
            # 只认紧接在记下的那次前向之后的交叉熵：同一次 compute_loss 里的 logits 和 loss
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
