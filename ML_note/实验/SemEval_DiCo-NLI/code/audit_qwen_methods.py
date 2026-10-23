"""Offline audit of three Qwen SMOKEs, formal CE/R-Drop and incomplete CE-drop package. Run with official repository path."""
import ast, csv, hashlib, json, math, statistics, sys
from pathlib import Path
from collections import Counter

EXP = Path(__file__).resolve().parent.parent
BASE = EXP / 'metrics/qwen_methods_v2'
OFF = Path(sys.argv[1]).resolve()
SRC = EXP / 'code/qwen_methods_v2/dico_qwen_methods.py'
sys.path.insert(0, str(OFF))
from evaluation_functions.scorer import score_files
from sklearn.metrics import f1_score

nodes = []
for n in ast.parse(SRC.read_text(encoding='utf8')).body:
    if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
        nodes.append(n)
    elif isinstance(n, ast.Assign):
        try: ast.literal_eval(n.value)
        except (ValueError, TypeError): continue
        nodes.append(n)
ns = {'__name__': 'audit', 'SMOKE': False}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), 'exec'), ns)
ns['DICO_DIR'] = str(OFF)
split = ns['make_split']()
sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
result = {'source_sha256': sha, 'checks': [], 'runs': {}, 'score_files': 0,
          'limits': 'Saved probabilities and scoring are independently checked; adapters and checkpoints were not supplied.'}
def read(p): return json.loads(p.read_text(encoding='utf8'))
def rows(p):
    with p.open(encoding='utf8', newline='') as f: return list(csv.DictReader(f))
def ck(name, ok):
    result['checks'].append({'name': name, 'passed': bool(ok)})
    assert ok, name
def finite(x):
    if isinstance(x, float): return math.isfinite(x)
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    return True
def valid_probs(p): return len(p) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p)-1) < 1e-5

for directory in sorted(p for p in BASE.iterdir() if p.is_dir() and p.name.endswith(('_results', '_smoke')) and (p/'fingerprint.json').exists()):
    fp, summary = read(directory/'fingerprint.json'), read(directory/'summary.json')
    method, smoke = fp['method'], fp['smoke']
    tag = directory.name
    ck(tag+' data identity', all(hashlib.sha256((OFF/p).read_bytes()).hexdigest() == h for p,h in fp['data_sha256'].items()))
    ck(tag+' method configuration', fp['method_params'] == ns['METHODS'][method])
    donefiles = sorted(directory.rglob('done.json'))
    if method == 'ce_drop' and not donefiles:
        ck(tag+' no completed seed or run metadata', not donefiles and not (directory/'run_meta.json').exists() and len(summary['missing']) == 3)
        continue
    meta = read(directory/'run_meta.json')
    ck(tag+' exact v2 hash and environment', len(meta)==1 and meta[0]['code_sha256']==sha and meta[0]['dtype']=='torch.bfloat16' and meta[0]['gpu']=='NVIDIA A10' and all(meta[0][k]==v for k,v in ns['PINS'].items()))
    ck(tag+' completed expected seeds', [p.parent.name for p in donefiles] == (['seed42','seed43'] if smoke else ['seed42','seed43','seed44']) and summary['missing']==[])
    parts = ns['track_parts'](1, split, smoke)
    families = [{split[0][r['pair_id']] for r in parts[k]} for k in ['fit','val','test']]
    ck(tag+' disjoint train validation holdout families', all(not families[i]&families[j] for i in range(3) for j in range(i)))
    if smoke:
        sc = read(directory/'smoke_checks.json')
        ck(tag+' all reported real SMOKE checks passed', all(sc['checks'].values()))
    scored = []
    for dp in donefiles:
        d, done = dp.parent, read(dp)
        name = tag+'/'+d.name
        history = read(d/'history.json')
        ck(name+' training records consistent and finite', done['history']==history and finite(done))
        ck(name+' rows and optimizer step count', done['fit_rows']==len(parts['fit']) and done['val_rows']==len(parts['val']) and done['steps']==ns['steps_per_epoch'](len(parts['fit']))*(2 if smoke else 3))
        ck(name+' forward precision agreement', done['forward_check']['max_letter_logprob_diff']==0 and done['forward_check']['argmax_agreement']==1 and done['forward_check']['autocast_ce_path']==done['forward_check']['autocast_hidden_path'])
        ck(name+' reported adapter reload exact', all(c['val_argmax_agreement']==1 and c['val_max_prob_diff']==0 for c in done['adapter_checks']))
        val_inputs = list(dict.fromkeys((r['text1'],r['text2']) for r in parts['val']))
        for h in history:
            ps=read(d/f"val_probs/epoch{h['epoch']}.json")
            ck(name+f" epoch{h['epoch']} validation probabilities valid", len(ps)==len(val_inputs) and all(valid_probs(p) for p in ps))
            by=dict(zip(val_inputs,ps))
            pred=[ns['LABELS'][max(range(4),key=lambda i:by[(r['text1'],r['text2'])][i])] for r in parts['val']]
            ck(name+f" epoch{h['epoch']} validation WF1", abs(f1_score([r['label'] for r in parts['val']],pred,average='weighted')-h['val_weighted_f1'])<1e-12)
        ck(name+' best epoch chosen only by inner validation', max(history,key=lambda h:h['val_weighted_f1'])['epoch']==done['best_epoch'])
        cache=[json.loads(line) for line in (d/'cache.jsonl').read_text(encoding='utf8').splitlines()]
        by={(r['a'],r['b']):r for r in cache}
        ck(name+' exact unique inference coverage',len(by)==len(cache) and set(by)==set(ns['needed_inputs'](parts['test']+parts['dev'])))
        ck(name+' probability and answer mass valid',all(valid_probs(r['p']) and math.isfinite(r['mass']) and 0<=r['mass']<=1.00001 for r in cache))
        for part,key in [('fold0','test'),('dev','dev')]:
            ref=d/f'reference_{part}.csv'
            if not ref.exists(): ref=OFF/'final_data/dev/dico_nli_dev_track1_reference.csv'
            official=rows(OFF/f"final_data/{'train' if key=='test' else 'dev'}/dico_nli_{'train' if key=='test' else 'dev'}_track1_reference.csv")
            expected={r['instance_id']:r for r in official if r['instance_id'] in {v['instance_id'] for v in parts[key]}}
            ck(name+'/'+part+' gold matches official subset',{r['instance_id']:{k:r[k] for k in ns['REFERENCE_FIELDS']} for r in rows(ref)}=={i:{k:r[k] for k in ns['REFERENCE_FIELDS']} for i,r in expected.items()})
            for mode in ['plain','sym']:
                pred=[]
                for r in parts[key]:
                    p=by[(r['text1'],r['text2'])]['p']
                    if mode=='sym':
                        q=by[(r['text2'],r['text1'])]['p'];p=[(p[i]+q[[0,2,1,3][i]])/2 for i in range(4)]
                    pred.append({'instance_id':r['instance_id'],'label':ns['LABELS'][max(range(4),key=p.__getitem__)]})
                pf=d/f'pred_{part}_{mode}.csv'
                ck(name+'/'+part+'/'+mode+' predictions derived from probabilities',rows(pf)==pred)
                score=score_files(ref,pf,write_outputs=False).to_dict()
                ck(name+'/'+part+'/'+mode+' official score reproduced',score==read(d/f'scores_{part}_{mode}/scores.json') and all(abs(score[k]-done['scores'][part][mode][k])<1e-12 for k in ns['METRICS']))
                result['score_files']+=1
        if smoke:
            ck(name+' expected interrupted/resumed seed',done['resumed_from_epoch']==(1 if d.name=='seed42' else 0))
        if method=='moco':
            stamps, step = [], 0
            # Independently simulate enqueue counts and FIFO age at each optimizer update.
            capacity = fp['method_params']['queue']
            for h in history:
                for start in range(0, len(parts['fit']), fp['train']['batch'] * fp['train']['grad_acc']):
                    step += 1
                    count = min(fp['train']['batch'] * fp['train']['grad_acc'], len(parts['fit']) - start)
                    stamps = (stamps + [step] * count)[-capacity:]
                age = sum(step - t for t in stamps) / len(stamps)
                ck(name+f" epoch{h['epoch']} queue and update counts", h['updates']==step and h['queue_filled']==len(stamps) and abs(h['queue_mean_key_age_updates']-age)<1e-4)
        result['runs'][name]={'scores':done['scores'],'best_epoch':done['best_epoch'],'peak_memory_gib':done['peak_memory_gib'],'seconds_per_step':done['seconds_per_step'],'minutes_last_session':done['minutes_last_session']}
        scored.append(done)
    for part in ['fold0','dev']:
        for mode in ['plain','sym']:
            for metric in ns['METRICS']:
                vals=[x['scores'][part][mode][metric] for x in scored]
                ck(tag+f' summary {part}/{mode}/{metric}',summary['results']['track1']['sft'][part+'_'+mode][metric]=={'mean':round(statistics.mean(vals),4),'sd':round(statistics.stdev(vals),4),'n':len(vals)})
result['passed']=all(c['passed'] for c in result['checks'])
(BASE/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'checks':len(result['checks']),'score_files':result['score_files'],'passed':result['passed']}))
