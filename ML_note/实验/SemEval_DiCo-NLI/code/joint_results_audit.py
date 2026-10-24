"""复核归档字节、31套OOF的62组官方分数，并记录KD逐条得失。"""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]/'metrics/joint_validation'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return list(csv.DictReader(p.open(encoding='utf-8')))

def main(official):
    sys.path.insert(0,str(Path(official).resolve()))
    from evaluation_functions import score_files
    manifest=read(ROOT/'ARCHIVE_MANIFEST.json')['files']
    selection=read(ROOT/'archive_selection.json')['included_files']
    for n in selection:
        if n=='ARCHIVE_MANIFEST.json': continue
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==manifest[n],n
    gold=Path(official)/'final_data/train/dico_nli_train_track1_reference.csv'
    complete=read(ROOT/'COMPLETE.json'); count=0; reports={}
    for pred in sorted((ROOT/'oof').rglob('pred_*.csv')):
        mode=pred.stem[5:]; key=pred.parent.relative_to(ROOT/'oof').as_posix()
        report=score_files(gold,pred)
        scores=report.to_dict()
        # score_files returns the official classification and consistency reports.
        stored=read(pred.parent/f'scores_{mode}.json')
        actual={'weighted_f1':report.classification.weighted_f1,
                'soft_cons':report.consistency.soft_cons,'hard_cons':report.consistency.hard_cons}
        for k,v in actual.items():
            assert abs(v-stored[k])<1e-10,(key,mode,k)
            assert abs(v-complete['oof'][key][mode][k])<1e-10,(key,mode,k)
        assert len(rows(pred))==3042
        reports[key+'/'+mode]=scores;count+=1
    assert count==62
    g={r['instance_id']:r['label'] for r in rows(gold)}
    changes={}
    for seed in [42,43,44]:
        maps={name:{r['instance_id']:r['label'] for r in rows(ROOT/'oof'/name/f'seed{seed}'/'pred_sym.csv')}
              for name in ['mdeberta/ce','kd','qwen']}
        a,b,t=maps['mdeberta/ce'],maps['kd'],maps['qwen']
        changes[str(seed)]={'corrected':sum(a[i]!=g[i] and b[i]==g[i] for i in g),
                            'new_errors':sum(a[i]==g[i] and b[i]!=g[i] for i in g),
                            'teacher_correct_student_wrong':sum(t[i]==g[i] and a[i]!=g[i] for i in g),
                            'kd_corrected_in_that_subset':sum(t[i]==g[i] and a[i]!=g[i] and b[i]==g[i] for i in g)}
    out={'archived_files_verified':len(selection)-1,'official_oof_scores_recomputed':count,
         'rows_per_oof':3042,'kd_sym_changes':changes,'official_reports':reports}
    (ROOT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='official_reports'},ensure_ascii=False))

if __name__=='__main__':main(sys.argv[1])
