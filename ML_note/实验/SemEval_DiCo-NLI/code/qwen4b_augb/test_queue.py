"""CPU 测试（不需要 torch）。RealData 一组需要官方数据：DICO_TEST_DATA=<解压后的官方仓库目录> python3 -m unittest test_queue"""
import tempfile, unittest, subprocess, sys, json, os, time, zipfile, copy
from pathlib import Path
import common
from common import *
from queue_augb import merge_predictions
Q=str(HERE/'queue_augb.py')

class Checks(unittest.TestCase):
    def test_identity_change(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'id.json'; bind(p,{'version':5})
            with self.assertRaises(ValueError): bind(p,{'version':4})
    def test_completed_artifact_damage(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); write(p/'done.json',{}); (p/'adapter_model.safetensors').write_bytes(b'test fixture')
            seal(p,{'pipeline':5}); verify(p,{'pipeline':5})
            with self.assertRaises(ValueError): verify(p,{'pipeline':4})
            (p/'adapter_model.safetensors').unlink()
            with self.assertRaises(ValueError): verify(p,{'pipeline':5})
    def test_oof_duplicate_missing(self):
        rows=[[{'instance_id':'a'}],[{'instance_id':'b'}]]
        self.assertEqual([r['instance_id'] for r in merge_predictions(rows,['b','a'])],['b','a'])
        with self.assertRaises(ValueError): merge_predictions(rows,['a'])
        with self.assertRaises(ValueError): merge_predictions(rows+rows,['a','b'])
    def test_probability(self):
        self.assertFalse(valid_p([float('nan'),0,0,1])); self.assertFalse(valid_p([1,1,0,0])); self.assertTrue(valid_p([.25]*4))
    def test_task_order(self):
        t1,t2=tasks(1),tasks(2)
        self.assertEqual(len(t1),30); self.assertEqual(len(set(t1)),30); self.assertEqual(len(t2),15)
        self.assertEqual(t1[:4],[('Q0',0,42),('Q-B1',0,42),('Q0',1,42),('Q-B1',1,42)])   # 同一折的 Q0 和 Q-B1 相邻
        self.assertEqual({g for g,_,_ in t2},{'Q-step1'})
    def test_source_identity(self):
        self.assertEqual(sha(HERE/'upstream_qwen.py'),UPSTREAM); self.assertEqual(len(pipeline()),64)
    def test_data_tamper_changes_identity(self):
        saved=dict(common.AUGB_DATA)
        try:
            common.AUGB_DATA['augb_data/manifest.csv']='0'*64
            with self.assertRaises(ValueError): common.pipeline()
        finally: common.AUGB_DATA.clear(); common.AUGB_DATA.update(saved)
    def test_plan_subprocess(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,Q,'plan','--root',t,'--data',t],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr); r=json.loads(p.stdout); self.assertEqual((len(r['stage1']),len(r['stage2'])),(30,15))
    def test_formal_rejects_without_confirmation(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,Q,'formal','--root',t,'--data',t],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0); self.assertIn('--confirm-formal',p.stderr)
            self.assertTrue((Path(t)/'download'/'qwen4b_augb_results.zip').exists())
    def test_stage2_gates(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,Q,'formal','--root',t,'--data',t,'--stage','2','--confirm-formal'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0); self.assertIn('--confirm-stage2',p.stderr)
            p=subprocess.run([sys.executable,Q,'formal','--root',t,'--data',t,'--stage','2','--confirm-formal','--confirm-stage2'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0); self.assertIn('阶段 1 还没全部完成',p.stderr)

class GeneratedRows(unittest.TestCase):
    """build_fit / check_generated 用随包的实验 B 数据；fit 用 split_manifest 的 id 构造假行，不需要官方数据。"""
    def fold_rows(self,k):
        fit=[{'instance_id':i,'pair_id':i.split('__')[0],'text1':'a'+i,'text2':'b'+i,'label':'EQUIVALENCE'} for i in sorted(augb_split(k)['fit'])]
        return fit,augb_generated(k)
    def test_counts_and_isolation_all_folds(self):
        for k in range(5):
            fit,gen=self.fold_rows(k); check_generated(k,fit,gen,True)
            q0,b1,st=(build_fit(g,fit,gen) for g in GROUPS)
            self.assertEqual((len(q0),len(b1),len(st)),(len(fit),2*len(fit),2*len(fit)))
            self.assertEqual(b1[:len(fit)],fit); self.assertEqual(st[:len(fit)],fit)
            self.assertTrue(all(r['instance_id'].startswith('gen:') for r in b1[len(fit):]))
            self.assertTrue(all(not any(x.startswith('_') for x in r) for r in b1))   # 内部字段不进训练行
            self.assertEqual([r['text1'] for r in st[len(fit):]],[r['text1'] for r in fit])
            self.assertEqual(len({r['instance_id'] for r in st}),len(st))
            self.assertEqual(len({fit_digest(q0),fit_digest(b1),fit_digest(st)}),3)
    def test_generated_label_mix(self):
        fit,gen=self.fold_rows(0)
        self.assertEqual({r['label'] for r in gen},{'EQUIVALENCE','FORWARD_ENTAILMENT','BACKWARD_ENTAILMENT','NEGATIVE_OTHER'})
    def test_isolation_rejects(self):
        fit,gen=self.fold_rows(0)
        bad=copy.deepcopy(gen); bad[0]['_context_parent_ids']+=';dico_phrasis_9999999__en-en__original'
        with self.assertRaises(ValueError): check_generated(0,fit,bad,True)
        bad=copy.deepcopy(gen); bad[0]['_usable_folds']=';'.join(x for x in '01234' if x!='0')
        with self.assertRaises(ValueError): check_generated(0,fit,bad,True)
        bad=copy.deepcopy(gen); bad[0]['_generator']='luna'
        with self.assertRaises(ValueError): check_generated(0,fit,bad,True)
        with self.assertRaises(ValueError): check_generated(0,fit,gen[:-1],True)
        with self.assertRaises(ValueError): check_generated(1,fit,gen,True)   # fold 0 的生成行拿到 fold 1 的 fit
        with self.assertRaises(ValueError): build_fit('Q-B2',fit,gen)

@unittest.skipUnless(os.environ.get('DICO_TEST_DATA'),'需要 DICO_TEST_DATA=<官方仓库目录>')
class RealData(unittest.TestCase):
    """用真实官方数据和未改动的上游 make_split：五折和实验 B 的划分逐折相同，三组的训练集按设计构造。"""
    def test_parts_all_folds(self):
        from worker import load, group_parts
        data=Path(os.environ['DICO_TEST_DATA']).resolve()
        with tempfile.TemporaryDirectory() as t:
            for k in range(5):
                q=load(Path(t),data,k,False,8)
                sizes={}
                for g in GROUPS:
                    parts,full,gen=group_parts(q,k,g,'formal'); sizes[g]=len(parts['fit'])
                    self.assertEqual(parts['val'],full['val']); self.assertEqual(parts['test'],full['test'])
                    self.assertEqual(parts['fit'][:len(full['fit'])],full['fit'])
                n=len(full['fit']); self.assertEqual(sizes,{'Q0':n,'Q-B1':2*n,'Q-step1':2*n})
                self.assertEqual(n,{0:2190,1:2192,2:2186,3:2200,4:2200}[k])
                if k==0:
                    q=load(Path(t),data,0,True,8)
                    sp,_,_=group_parts(q,0,'Q-B1','smoke'); s0,_,_=group_parts(q,0,'Q0','smoke')
                    self.assertEqual(len(sp['fit']),2*len(s0['fit'])); self.assertEqual(sp['val'],full['val'])
                    pp,_,_=group_parts(q,0,'Q-B1','probe'); self.assertEqual(len(pp['fit']),2*n)

class Package(unittest.TestCase):
    def _official(self,t):
        z=Path(t)/'official.zip'
        with zipfile.ZipFile(z,'w') as f: f.writestr('Repo-main/final_data/train/dico_nli_train_track1_participant_labeled.csv','x')
        return z
    def _adapter(self,t):
        a=Path(t)/'ad'; a.mkdir(); (a/'adapter_model.safetensors').write_bytes(b'fixture'); (a/'adapter_config.json').write_text('{}'); return a
    def test_upload_whitelist(self):
        import prepare_upload
        stray=HERE/'stray_old_package.zip'; stray.write_bytes(b'should not ship')
        try:
            with tempfile.TemporaryDirectory() as t:
                off,ad=self._official(t),self._adapter(t)
                for out in (HERE/'a10_upload_test_inside.zip', Path(t)/'outside.zip'):
                    try:
                        prepare_upload.build(off,ad,out); names=set(zipfile.ZipFile(out).namelist())
                        allowed={'qwen4b_augb/'+n for n in ship_files()}|{'inputs/zero4b_cache.jsonl','official.zip','inputs/adapter4b/adapter_model.safetensors','inputs/adapter4b/adapter_config.json','UPLOAD_MANIFEST.json'}
                        self.assertEqual(names,allowed)
                        self.assertTrue({'qwen4b_augb/'+n for n in AUGB_DATA}<=names)
                        self.assertFalse(Path(str(out)+'.tmp').exists())
                    finally:
                        if out.exists(): out.unlink()
        finally: stray.unlink()
    def test_second_coordinator_rejected_and_pack_contents(self):
        with tempfile.TemporaryDirectory() as t:
            env={**os.environ,'DICO_TEST_HOLD_SECONDS':'4'}
            cmd=[sys.executable,Q,'pack','--root',t,'--data',t]
            first=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            time.sleep(1.5)
            second=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0); self.assertIn('正在运行',second.stderr)
            out,err=first.communicate(timeout=60); self.assertEqual(first.returncode,0,err)
            names=zipfile.ZipFile(Path(t)/'download'/'qwen4b_augb_results.zip').namelist()
            self.assertFalse(any(n.endswith('.queue.lock') for n in names))
            for n in ('source/baseline/zero4b_cache_a10_t1.jsonl','source/prepare_upload.py','source/augb_data/manifest.csv'): self.assertIn(n,names)
    def test_receipts_readonly_on_fresh_root(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,Q,'receipts','--root',t,'--data',t],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr); r=json.loads(p.stdout)
            self.assertEqual((r['SMOKE'],r['COMPATIBLE'],r['stage1_complete'],r['stage2_complete']),('还没有','还没有','0 / 30','0 / 15'))
            self.assertEqual(sorted(x.name for x in Path(t).iterdir()),[])
    def test_receipts_rejects_fake_files(self):
        with tempfile.TemporaryDirectory() as t:
            write(Path(t)/'SMOKE.json',{'smoke_passed':True}); write(Path(t)/'COMPATIBLE.json',{'passed':True})
            r=json.loads(subprocess.run([sys.executable,Q,'receipts','--root',t,'--data',t],capture_output=True,text=True).stdout)
            self.assertTrue(r['SMOKE'].startswith('没通过')); self.assertTrue(r['COMPATIBLE'].startswith('没通过'))
    def test_download_timeout_two_hours(self):
        src=(HERE/'queue_augb.py').read_text(); self.assertEqual(src.count("timeout=7200 if kind=='download'"),2)
    def test_both_downloads_use_fallback(self):
        for n in ('worker.py','compat.py'):
            s=(HERE/n).read_text(); self.assertNotIn('q.fetch_from_modelscope(',s.replace('try: return q.fetch_from_modelscope(rev)',''))
if __name__=='__main__': unittest.main()
