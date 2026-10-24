"""Offline regression/injection tests; synthetic outputs are never research evidence."""
import contextlib
import copy
import io
import shutil
import tempfile
import unittest
import types
import sys
from unittest.mock import patch
import joint_common as c
import joint_queue as j
import joint_worker as w

ROOT=c.HERE.parent/'tmp'/'joint_test'

class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q,cls.m=c.modules(ROOT)

    def test_original_sources_exact(self):
        self.assertEqual(c.sha(c.HERE/'dico_qwen_methods.py'),c.QSHA)
        self.assertEqual(c.sha(c.HERE/'dico_mdeberta_methods.py'),c.MSHA)

    def test_five_splits_and_oof(self):
        allids=[]
        for fold in range(5):
            q,m=c.get_parts(self.q,self.m,fold)
            self.assertEqual(c.signature(q),c.signature(m))
            allids += [r['instance_id'] for r in q['test']]
            qs,ms=c.get_parts(self.q,self.m,fold,True)
            self.assertEqual(c.signature(qs),c.signature(ms))
        self.assertEqual(len(allids),3042); self.assertEqual(len(set(allids)),3042)

    def test_original_mdeberta_restore_and_skip(self):
        m=self.m; old=m.OUT; m.OUT=str(ROOT/'mdeberta'/'dico_mdeberta_results')
        try:
            tasks=m.stage_tasks('stage2'); self.assertEqual(len(tasks),125)
            done=[]
            for t in tasks:
                p=c.Path(m.task_dir(t))
                if not (p/'done.json').exists(): continue
                before=c.read(p/'task.json'); expected=m.task_fingerprint(t,c.MREV)
                # Existing records were produced on Linux, while this test can run on Windows.
                expected['data_sha256']={k.replace('\\','/'):v for k,v in expected['data_sha256'].items()}
                self.assertEqual(before,expected); done.append(t)
            self.assertEqual(len(done),16)
        finally: m.OUT=old

    def test_original_qwen_fold0_fingerprint_and_weights(self):
        q=self.q; q.HOLDOUT_FOLD=0
        src=c.qout(ROOT,0)
        fp=q.fingerprint(); fp['data_sha256']={k.replace('\\','/'):v for k,v in fp['data_sha256'].items()}
        self.assertEqual(q.incompatible(c.read(src/'fingerprint.json'),fp),[])
        wm=c.read(src/'weights_manifest.json')
        for a in wm['adapters']:
            d=src/a['run']; self.assertEqual(c.read(d/'done.json')['best_epoch'],a['best_epoch'])
            for n,h in a['sha256'].items(): self.assertEqual(c.sha(d/'adapters'/f"epoch{a['best_epoch']}"/n),h)

    def test_qwen_holdout_scores_match_original(self):
        q=self.q; parts,_=c.get_parts(q,self.m,0)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d=c.Path(temp); source=c.qout(ROOT,0)/'track1/seed42'
            shutil.copyfile(source/'cache.jsonl',d/'cache.jsonl')
            class Backend:
                def score(self,inputs): raise AssertionError('complete cache should not infer again')
            scores,stats=w.score_qwen(q,Backend(),d,parts)
            old=c.read(source/'done.json')['scores']
            self.assertEqual(scores['holdout'],old['fold0']); self.assertEqual(scores['dev'],old['dev'])
            self.assertEqual(stats['scored_this_session'],0)

    def fixture_teacher(self,root,fold=2,seed=42):
        q=self.q; parts,_=c.get_parts(q,self.m,fold,True)
        src=c.qout(root,fold,True); run=src/'track1'/f'seed{seed}'
        c.write(src/'fingerprint.json',{'model_revision':c.QREV,'split':{'holdout_fold':fold},'smoke':True,'method':'ce'})
        c.write(run/'done.json',{'best_epoch':1})
        ad=run/'adapters/epoch1/adapter_model.safetensors'; ad.parent.mkdir(parents=True); ad.write_bytes(b'SYNTHETIC-NOT-MODEL')
        d=c.tout(root,fold,seed,True)
        c.write(d/'identity.json',{'fold':fold,'seed':seed,'smoke':True,'pipeline_id':c.pipeline_id(),'labels':c.LABELS,'source_training_sha256':c.QSHA,
                 'split':c.signature(parts),'best_epoch':1,'adapter_sha256':c.sha(ad),
                 'source_fingerprint_sha256':c.sha(src/'fingerprint.json'),'source_done_sha256':c.sha(run/'done.json')})
        with (d/'teacher.jsonl').open('w',encoding='utf-8') as f:
            for key,part in c.expected_inputs(q,parts).items():
                a,b=c.json.loads(key); f.write(c.json.dumps({'a':a,'b':b,'part':part,'p':[.1,.2,.3,.4]},ensure_ascii=False)+'\n')
        done={'identity_sha256':c.sha(d/'identity.json'),'teacher_sha256':c.sha(d/'teacher.jsonl'),
              'checks':{'val_argmax_agreement':1.,'cache_argmax_agreement':1.}}
        c.write(d/'done.json',done)
        return d,parts

    def test_teacher_identity_and_mapping(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d,parts=self.fixture_teacher(temp)
            probs,h=c.validate_teacher(self.q,temp,2,42,parts,True)
            self.assertEqual(len(probs),len(parts['fit'])); self.assertEqual(probs[0],[.1,.2,.3,.4])
            for field,value in [('fold',1),('seed',43),('smoke',False),('labels',list(reversed(c.LABELS))),('pipeline_id','wrong')]:
                original=c.read(d/'identity.json'); bad=copy.deepcopy(original); bad[field]=value
                c.write(d/'identity.json',bad)
                with self.assertRaises(ValueError): c.validate_teacher(self.q,temp,2,42,parts,True)
                c.write(d/'identity.json',original)
            shuffled=copy.deepcopy(parts); shuffled['fit']=list(reversed(parts['fit']))
            with self.assertRaises(ValueError): c.validate_teacher(self.q,temp,2,42,shuffled,True)

    def test_teacher_duplicate_leak_nonfinite_missing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d,parts=self.fixture_teacher(temp); p=d/'teacher.jsonl'; original=p.read_text(encoding='utf-8'); lines=original.splitlines()
            cases=[original+lines[0]+'\n','\n'.join(lines[1:])+'\n']
            for field,value in [('part','test'),('p',[float('nan'),.2,.3,.4]),('p',[.1,.2,.3,.5]),('a','outside fit')]:
                x=c.json.loads(lines[0]); x[field]=value; cases.append(c.json.dumps(x)+'\n'+'\n'.join(lines[1:])+'\n')
            for bad in cases:
                p.write_text(bad,encoding='utf-8')
                with self.assertRaises(ValueError): c.teacher_rows(self.q,p,parts)
            p.write_text(original,encoding='utf-8')
            c.teacher_rows(self.q,p,parts)

    def test_truncated_teacher_cache_recovery(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d,parts=self.fixture_teacher(temp); p=d/'teacher.jsonl'
            with p.open('a') as f:f.write('{"a":')
            self.q.load_jsonl(str(p)); c.teacher_rows(self.q,p,parts)
            self.assertEqual(len(list(d.glob('*.broken-*'))),1)

    def test_export_and_resume_with_fake_model_real_tensor_checks(self):
        import torch
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            q=self.q; fold=2; seed=42; parts,_=c.get_parts(q,self.m,fold,True)
            oldout=q.OUT; q.OUT=str(c.qout(temp,fold,True)); src=c.Path(q.OUT); run=src/'track1/seed42'
            c.write(src/'fingerprint.json',{'model_revision':c.QREV,'split':{'holdout_fold':fold},'smoke':True,'method':'ce'})
            c.write(run/'done.json',{'best_epoch':1})
            adapter=run/'adapters/epoch1/adapter_model.safetensors'; adapter.parent.mkdir(parents=True); adapter.write_bytes(b'fake-model')
            vi=list(dict.fromkeys((r['text1'],r['text2']) for r in parts['val']))
            c.write(run/'val_probs/epoch1.json',[[.1,.2,.3,.4] for _ in vi])
            q.append_jsonl(str(run/'cache.jsonl'),[{'a':a,'b':b,'p':[.1,.2,.3,.4]} for a,b in q.needed_inputs(parts['test'])])
            tensors={'test':torch.arange(4,dtype=torch.float32)}
            class Model:
                def eval(self):pass
            class Backend:
                model=None
                reloads=0
                calls=0
                def train(self,*args):self.reloads+=1; self.model=Model()
                def score(self,items):self.calls+=1; return [{'p':[.1,.2,.3,.4],'mass':.9,'top_is_letter':True} for _ in items]
            b=Backend()
            fake_peft=types.SimpleNamespace(get_peft_model_state_dict=lambda model:tensors)
            fake_safe=types.SimpleNamespace(load_file=lambda path:tensors)
            try:
                with patch.dict(sys.modules,{'peft':fake_peft,'safetensors.torch':fake_safe}):
                    w.export_teacher(q,b,temp,fold,seed,parts,True)
                    self.assertEqual(b.reloads,1); calls=b.calls
                    w.export_teacher(q,b,temp,fold,seed,parts,True)
                    self.assertEqual(b.calls,calls)
                probs,_=c.validate_teacher(q,temp,fold,seed,parts,True)
                self.assertEqual(len(probs),len(parts['fit']))
            finally:q.OUT=oldout

    def test_kd_worker_integration_and_skip_with_fake_training(self):
        import torch
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            self.fixture_teacher(temp)
            calls=[]
            def load(root,smoke):
                q,m=c.modules(ROOT,smoke)
                m.ensure_packages=lambda:None
                m.fetch_model=lambda rev:('fake',c.MREV,'test')
                m.amp_dtype=lambda:torch.bfloat16
                m.env_info=lambda:{'gpu':'synthetic','torch':torch.__version__}
                def train(task,parts,teacher=None,**kw):
                    calls.append(task)
                    self.assertEqual(len(teacher),len(parts['fit']))
                    self.assertEqual(teacher[0],[.1,.2,.3,.4])
                    logits=lambda rs:{x:[1.,0.,0.,0.] for x in m.needed_inputs(rs)}
                    return {'val_logits':logits(parts['val']),'test_logits':logits(parts['test']),'dev_logits':logits(parts['dev']),
                            'history':[{'epoch':1,'val_weighted_f1':0.1}],'best_epoch':1,'val_weighted_f1':0.1,
                            'precision':'bf16 autocast','precision_attempts':[],'updates':1,'skipped_updates':0,
                            'peak_memory_gib':0,'peak_reserved_gib':0,'seconds_per_step':0,'steps':1,'steps_per_epoch':1}
                m.TRAIN_FN=train
                return q,m
            with patch.object(w,'modules',side_effect=load),patch.object(w,'cache_fetch'),patch.object(torch.cuda,'is_available',return_value=True):
                w.run_kd(temp,2,42,True)
                w.run_kd(temp,2,42,True)
            self.assertEqual(len(calls),1)
            task=c.read(c.kdout(temp,True)/'track1/kd/seed42/fold2/task.json')
            self.assertEqual(task['smoke_rows']['fit'],192)
            self.assertTrue(task['with_dev'])

    def test_queue_failure_blocks_only_dependent_kd(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            c.write(c.Path(temp)/'SMOKE_OK.json',{'pipeline_id':c.pipeline_id()})
            seen=[]
            def call(root,label,args,env=None):
                seen.append(label); return label!='qwen_fold2_seed43'
            with patch.object(j,'call',side_effect=call),patch.object(j,'pack'),patch.object(j,'budget'),patch.object(j,'validate') as val:
                with self.assertRaises(ValueError):j.run(temp)
                val.assert_not_called()
            self.assertEqual(len(seen),30); self.assertNotIn('kd_fold2_seed43',seen)
            self.assertIn('kd_fold4_seed44',seen)

    def test_smoke_failure_never_runs_formal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            with patch.object(j,'call',return_value=False) as call,patch.object(j,'pack'):
                with self.assertRaises(RuntimeError):j.smoke(temp)
                self.assertEqual(call.call_count,1)
            self.assertFalse((c.Path(temp)/'SMOKE_OK.json').exists())

    def test_budget_gate_before_formal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=c.Path(temp)
            p=c.qout(root,1,True)/'track1/seed43/done.json'
            c.write(p,{'seconds_per_step':2.0})
            c.write(c.tout(root,1,43,True)/'done.json',{'minutes':.01,'inputs':200})
            with patch.object(j,'modules',return_value=(self.q,self.m)):
                j.budget(root)
                self.assertLessEqual(c.read(root/'budget_estimate.json')['estimated_session_calculation'],78)
                c.write(p,{'seconds_per_step':100.})
                with self.assertRaises(ValueError):j.budget(root)

    def test_partial_results_never_complete(self):
        with self.assertRaises(FileNotFoundError): j.validate(ROOT)
        self.assertFalse((ROOT/'COMPLETE.json').exists())

    def test_complete_synthetic_matrix_and_corruption_gate(self):
        # Exercise positive final assembly using synthetic constant predictions, not model results.
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=c.Path(temp); shutil.copytree(ROOT/'official',root/'official')
            q,m=c.modules(root); sys.path.insert(0,str(root/'official'))
            md=root/'mdeberta/dico_mdeberta_results'; c.write(md/'moco_m_choice.json',{'m':.999})
            original_cwd=q.DICO_DIR
            for fold in range(5):
                parts,mp=c.get_parts(q,m,fold)
                score={mode:j.check_scores(root,q,parts['test'],[c.LABELS[0]]*len(parts['test']),root/'fixture_scores'/f'fold{fold}',mode) for mode in ('plain','sym')}
                dev={mode:j.check_scores(root,q,parts['dev'],[c.LABELS[0]]*len(parts['dev']),root/'fixture_scores'/f'dev{fold}',mode) for mode in ('plain','sym')}
                lm={x:[1.,0.,0.,0.] for x in m.needed_inputs(mp['test'])}
                for method in j.METHODS:
                    for seed in range(42,47):
                        d=md/f'track1/{method}/seed{seed}/fold{fold}'
                        c.write(d/'task.json',m.task_fingerprint(m.make_task(1,method,seed,fold,fold==0),c.MREV))
                        c.write(d/'done.json',{'code_sha256':c.MSHA,'precision':'bf16 autocast','skipped_updates':0,'scores':{'test':score}})
                        m.save_logits(str(d/'logits_test.json'),lm)
                src=c.qout(root,fold); fp=q.fingerprint(); fp['model_revision']=c.QREV
                c.write(src/'fingerprint.json',fp); c.write(src/'run_meta.json',[{'dtype':'torch.bfloat16','code_sha256':c.QSHA}])
                c.write(src/'joint_driver.json',{'pipeline_id':c.pipeline_id()})
                for seed in (42,43,44):
                    run=src/'track1'/f'seed{seed}'
                    c.write(run/'done.json',{'best_epoch':1,'history':[{'epoch':1,'val_weighted_f1':.5}],'scores':{'holdout':score,'dev':dev}})
                    q.append_jsonl(str(run/'cache.jsonl'),[{'a':a,'b':b,'p':[1.,0.,0.,0.]} for a,b in q.needed_inputs(parts['test']+parts['dev'])])
                    ad=run/'adapters/epoch1/adapter_model.safetensors'; ad.parent.mkdir(parents=True); ad.write_bytes(b'SYNTHETIC-NOT-MODEL')
                    td=c.tout(root,fold,seed)
                    c.write(td/'identity.json',{'fold':fold,'seed':seed,'smoke':False,'pipeline_id':c.pipeline_id(),'labels':c.LABELS,'source_training_sha256':c.QSHA,
                         'split':c.signature(parts),'best_epoch':1,'adapter_sha256':c.sha(ad),
                         'source_fingerprint_sha256':c.sha(src/'fingerprint.json'),'source_done_sha256':c.sha(run/'done.json')})
                    with (td/'teacher.jsonl').open('w',encoding='utf-8') as f:
                        for key,part in c.expected_inputs(q,parts).items():
                            a,b=c.json.loads(key); f.write(c.json.dumps({'a':a,'b':b,'part':part,'p':[1.,0.,0.,0.]},ensure_ascii=False)+'\n')
                    th=c.sha(td/'teacher.jsonl')
                    c.write(td/'done.json',{'teacher_sha256':th,'identity_sha256':c.sha(td/'identity.json'),
                                          'checks':{'val_argmax_agreement':1.,'cache_argmax_agreement':1.}})
                    kd=c.kdout(root)/f'track1/kd/seed{seed}/fold{fold}'
                    kfp=m.task_fingerprint(m.make_task(1,'kd',seed,fold,fold==0),c.MREV,{'teacher_sha256':th})
                    kfp['joint_pipeline_id']=c.pipeline_id(); kfp['method_params']['source_training_sha256']=c.QSHA
                    c.write(kd/'task.json',kfp)
                    c.write(kd/'done.json',{'code_sha256':c.pipeline_id(),'precision':'bf16 autocast','skipped_updates':0,'scores':{'test':score}})
                    m.save_logits(str(kd/'logits_test.json'),lm)
            result=j.validate(root)
            self.assertEqual(result['oof_count'],31); self.assertEqual(result['counts']['mdeberta'],125)
            p=c.kdout(root)/'track1/kd/seed44/fold4/task.json'; bad=c.read(p); bad['teacher_sha256']='wrong'; c.write(p,bad)
            with self.assertRaises(ValueError):j.validate(root)
            self.assertFalse((root/'COMPLETE.json').exists())

    def test_zip_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=c.Path(temp)/'bad.zip'
            with c.zipfile.ZipFile(p,'w') as z:z.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError):c.extract(p,c.Path(temp)/'output')

    def test_result_and_weight_pack_separate_and_hashed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            r=c.Path(temp); c.write(r/'formal/qwen/fold0/done.json',{'synthetic':True})
            ad=r/'formal/qwen/fold0/adapters/epoch1/fake.safetensors'; ad.parent.mkdir(parents=True); ad.write_bytes(b'test')
            ck=r/'formal/qwen/fold0/checkpoints/x.bin'; ck.parent.mkdir(); ck.write_bytes(b'checkpoint')
            j.pack(r)
            for kind in ('results','weights'):
                with c.zipfile.ZipFile(r/'download'/f'dico_joint_{kind}.zip') as z:
                    man=c.json.loads(z.read('ARCHIVE_MANIFEST.json'))
                    for n,h in man['files'].items():self.assertEqual(c.hashlib.sha256(z.read(n)).hexdigest(),h)
                    self.assertFalse(any('checkpoints' in n for n in z.namelist()))
                    self.assertEqual(any('safetensors' in n for n in z.namelist()),kind=='weights')

if __name__=='__main__': unittest.main(verbosity=2)
