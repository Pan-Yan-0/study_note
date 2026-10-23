"""Offline audit of actual A10 v3 stage1; no downloads or model training.

Reads immutable copies under review_group5_smoke_20261022. The training module
is AST-loaded without running its top-level environment, installation or main.
"""
import ast
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
PKG = ROOT / 'mdeberta_methods_v3'
OUT = EXP / 'metrics/mdeberta_methods_stage1'
OFF = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(OFF))
from evaluation_functions.scorer import score_files


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    with path.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


source = PKG / 'dico_mdeberta_methods.py'
sha = hashlib.sha256(source.read_bytes()).hexdigest()
nodes = []
for node in ast.parse(source.read_text(encoding='utf8')).body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
        nodes.append(node)
    elif isinstance(node, ast.Assign):
        try:
            ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        nodes.append(node)
ns = {'__name__': 'audit', 'SMOKE': True}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
ns.update(DICO_DIR=str(OFF), L2I={x: i for i, x in enumerate(ns['LABELS'])})
report = {'source_sha256': sha, 'checks': [], 'runs': {},
          'scope': 'Offline saved-output audit ; no pretrained model rerun',
          'local_torch': torch.__version__}


def check(name, value):
    report['checks'].append({'name': name, 'passed': bool(value)})
    if not value:
        raise AssertionError(name)


def finite_tree(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, list):
        return all(finite_tree(v) for v in x)
    return True


meta = read(OUT / 'run_meta.json')
check('Exact source matches actual smoke hash', all(m['code_sha256'] == sha for m in meta))
check('Actual A10 bf16 run', meta[0]['gpu'] == 'NVIDIA A10' and meta[0]['amp'] == 'bf16')
summary = read(OUT / 'summary.json')
done_paths = sorted(OUT.rglob('done.json'))
check('30 stage1 tasks plus first completed stage2 task', len(done_paths) == 31 and len(summary['runs']) == 30)
attempts = [json.loads(s) for s in (OUT / 'attempts.jsonl').read_text(encoding='utf8').splitlines()]
report['attempt_events'] = dict(Counter(x['event'] for x in attempts))
report['score_files_checked'] = 0
report['logit_values_checked'] = 0

for dp in done_paths:
    d = dp.parent
    name = d.relative_to(OUT).as_posix()
    done, task, history = read(dp), read(d / 'task.json'), read(d / 'history.json')
    parts = ns['parts_for'](task['track'], task['fold'], task['with_dev'], smoke=False)
    full = ns['parts_for'](task['track'], task['fold'], task['with_dev'], smoke=False)
    fam, _ = ns['split_base']()
    sets = [{fam[r['pair_id']] for r in full[k]} for k in ['fit', 'val', 'test']]
    check(name + ' family disjoint fit/val/heldout', all(not sets[i] & sets[j] for i in range(3) for j in range(i)))
    check(name + ' official data hashes', all(hashlib.sha256((OFF / p).read_bytes()).hexdigest() == h for p, h in task['data_sha256'].items()))
    check(name + ' row counts match reproducible subset', done['rows'] == {k: len(v) for k, v in parts.items()})
    check(name + ' code and precision identity', done['code_sha256'] == sha and done['precision_attempts'] == [{'precision': 'bf16 autocast', 'result': 'ok'}])
    step = math.ceil(len(parts['fit']) / task['train']['batch'])
    check(name + ' expected successful optimizer updates', done['updates'] == 12 * step and done['skipped_updates'] == 0 and len(history) == 12 and all(h['updates'] == step and h['skipped_updates'] == 0 for h in history))
    check(name + ' finite training diagnostics', finite_tree(history) and all(h['grad_norm_mean'] > 0 for h in history))
    best = max(history, key=lambda h: h['val_weighted_f1'])
    check(name + ' earliest best inner validation epoch', done['best_epoch'] == best['epoch'] and done['val_weighted_f1'] == best['val_weighted_f1'])
    check(name + ' completed attempt recorded', any(a['event'] == 'completed' and a['attempt_id'] == done['attempt_id'] for a in attempts))
    rr = {'updates': done['updates'], 'peak_memory_gib': done['peak_memory_gib'],
          'peak_reserved_gib': done['peak_reserved_gib'], 'best_epoch': done['best_epoch'],
          'rows': done['rows'], 'predictions': {}, 'losses': [h['train_loss'] for h in history]}
    for part in [k for k in ['val', 'test', 'dev'] if k in parts]:
        saved = read(d / f'logits_{part}.json')
        keys = [tuple(x) for x in saved['inputs']]
        values = np.array(saved['logits'], dtype=np.float64)
        expected = set(ns['needed_inputs'](parts[part], both=part != 'val'))
        check(name + '/' + part + ' exact unique input coverage', len(keys) == len(set(keys)) and set(keys) == expected)
        check(name + '/' + part + ' finite four class logits', values.shape == (len(keys), 4) and bool(np.isfinite(values).all()))
        report['logit_values_checked'] += values.size
        exp = np.exp(values - values.max(axis=1, keepdims=True))
        pmap = dict(zip(keys, exp / exp.sum(axis=1, keepdims=True)))
        check(name + '/' + part + ' logits depend on input', bool(np.any(values != values[0])))
        for mode in (['plain'] if part == 'val' else ['plain', 'sym']):
            pred = []
            for r in parts[part]:
                p = pmap[(r['text1'], r['text2'])]
                if mode == 'sym':
                    p = (p + pmap[(r['text2'], r['text1'])][[0, 2, 1, 3]]) / 2
                pred.append(ns['LABELS'][int(np.argmax(p))])
            wf1 = f1_score([r['label'] for r in parts[part]], pred, average='weighted', zero_division=0)
            if part == 'val':
                check(name + ' saved best logits reproduce inner WF1', abs(wf1 - done['val_weighted_f1']) < 1e-12)
                continue
            predfile = d / f'pred_{part}_{mode}.csv'
            actual = rows(predfile)
            check(name + f'/{part}/{mode} predictions derive from logits', actual == [{'instance_id': r['instance_id'], 'label': p} for r, p in zip(parts[part], pred)])
            reference = rows(d / f'reference_{part}.csv')
            check(name + '/' + part + ' gold equals official subset', reference == [{k: r[k] for k in ns['REFERENCE_FIELDS']} for r in parts[part]])
            score = score_files(d / f'reference_{part}.csv', predfile, write_outputs=False).to_dict()
            check(name + f'/{part}/{mode} full official score reproduction', score == read(d / f'scores_{part}_{mode}/scores.json'))
            check(name + f'/{part}/{mode} done scores and independent sklearn WF1', all(abs(score[k] - done['scores'][part][mode][k]) < 1e-12 for k in ns['METRICS']) and abs(wf1 - score['weighted_f1']) < 1e-12)
            rr['predictions'][part + '_' + mode] = dict(Counter(pred))
            report['score_files_checked'] += 1
    if task['method'].startswith('moco'):
        check(name + ' queue grew with successful batches', [h['queue_filled'] for h in history] == [min(4096, len(parts['fit']) * ep) for ep in range(1,13)] and all(h['queue_mean_key_age_updates'] >= 0 for h in history))
    if name in summary['runs']:
        check(name + ' summary agrees with done', all(v == done[k] for k, v in summary['runs'][name].items()))
    report['runs'][name] = rr


choice=read(OUT/'moco_m_choice.json')
means={str(m):sum(read(OUT/f'track1/moco_m{m}/seed{s}/fold0/done.json')['val_weighted_f1'] for s in [42,43,44])/3 for m in [.99,.999]}
check('MoCo chosen solely by inner mean', means==choice['inner_val_wf1_mean'] and choice['m']==float(max(means,key=means.get)))
report['all_passed']=all(x['passed'] for x in report['checks'])
dest=OUT/'independent_audit.json'
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'checks':len(report['checks']),'scores':report['score_files_checked'],'logits':report['logit_values_checked'],'passed':report['all_passed']},ensure_ascii=False))
