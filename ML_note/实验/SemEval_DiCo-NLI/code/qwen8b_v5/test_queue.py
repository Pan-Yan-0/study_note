import tempfile, unittest, subprocess, sys, json, os, time, zipfile
from pathlib import Path
from common import *
from a1_queue import merge_predictions
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
    def test_seed_major(self):
        self.assertEqual(tasks()[:5],[(i,42) for i in range(5)]); self.assertEqual(len(set(tasks())),15)
    def test_plan_subprocess(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,str(HERE/'a1_queue.py'),'plan','--root',t,'--data',t],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr); self.assertEqual(len(json.loads(p.stdout)['tasks']),15)
    def test_formal_rejects_without_confirmation(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,str(HERE/'a1_queue.py'),'formal','--root',t,'--data',t],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0); self.assertIn('--confirm-formal',p.stderr)
            self.assertTrue((Path(t)/'download'/'a1_qwen8b_v5_results.zip').exists())
    def test_source_identity(self): self.assertEqual(sha(HERE/'upstream_qwen.py'),UPSTREAM); self.assertEqual(len(pipeline()),64)
class V52(unittest.TestCase):
    def _official(self,t):
        z=Path(t)/'official.zip'
        with zipfile.ZipFile(z,'w') as f: f.writestr('Repo-main/final_data/train/dico_nli_train_track1_participant_labeled.csv','x')
        return z
    def _adapter(self,t):
        a=Path(t)/'ad'; a.mkdir(); (a/'adapter_model.safetensors').write_bytes(b'fixture'); (a/'adapter_config.json').write_text('{}'); return a
    def test_upload_never_contains_itself(self):
        import prepare_upload
        stray=HERE/'stray_old_package.zip'; stray.write_bytes(b'should not ship')
        try:
            with tempfile.TemporaryDirectory() as t:
                for out in (HERE/'a10_upload_test_inside.zip', Path(t)/'outside.zip'):
                    try:
                        prepare_upload.build(self._official(t) if not (Path(t)/'official.zip').exists() else Path(t)/'official.zip', self._adapter(t) if not (Path(t)/'ad').exists() else Path(t)/'ad', out)
                        names=set(zipfile.ZipFile(out).namelist())
                        self.assertFalse(any(n.endswith(out.name) for n in names), names)
                        self.assertFalse(any('stray_old_package' in n for n in names))
                        allowed={'qwen8b_v5/'+n for n in ship_files()}|{'inputs/zero4b_cache.jsonl','official.zip','inputs/adapter4b/adapter_model.safetensors','inputs/adapter4b/adapter_config.json','UPLOAD_MANIFEST.json'}
                        self.assertEqual(names,allowed)
                        self.assertFalse(Path(str(out)+'.tmp').exists())
                    finally:
                        if out.exists(): out.unlink()
        finally: stray.unlink()
    def test_upload_without_adapter_is_marked(self):
        import prepare_upload
        with tempfile.TemporaryDirectory() as t:
            out=prepare_upload.build(self._official(t),None,Path(t)/'u.zip'); names=zipfile.ZipFile(out).namelist()
            self.assertTrue(any(n.startswith('inputs/adapter4b/') and n.endswith('.txt') for n in names))
            self.assertFalse(any(n.endswith('adapter_model.safetensors') for n in names))
    def test_second_coordinator_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            env={**os.environ,'DICO_TEST_HOLD_SECONDS':'4'}
            cmd=[sys.executable,str(HERE/'a1_queue.py'),'pack','--root',t,'--data',t]
            first=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            time.sleep(1.5)
            second=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0); self.assertIn('正在运行',second.stderr)
            out,err=first.communicate(timeout=60); self.assertEqual(first.returncode,0,err)
            names=zipfile.ZipFile(Path(t)/'download'/'a1_qwen8b_v5_results.zip').namelist()
            self.assertFalse(any(n.endswith('.queue.lock') for n in names))
            self.assertIn('source/baseline/zero4b_cache_a10_t1.jsonl',names)   # R4：源码快照能重建默认上传包
            self.assertIn('source/prepare_upload.py',names)
    def test_receipts_readonly_on_fresh_root(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,str(HERE/'a1_queue.py'),'receipts','--root',t,'--data',t],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr); r=json.loads(p.stdout)
            self.assertEqual(r['SMOKE'],'还没有'); self.assertEqual(r['COMPATIBLE'],'还没有'); self.assertEqual(r['formal_complete'],'0 / 15')
            self.assertEqual(sorted(x.name for x in Path(t).iterdir()),[])   # 只读：不建任何文件
    def test_receipts_rejects_fake_files(self):
        with tempfile.TemporaryDirectory() as t:
            write(Path(t)/'SMOKE.json',{'smoke_passed':True}); write(Path(t)/'COMPATIBLE.json',{'passed':True})
            p=subprocess.run([sys.executable,str(HERE/'a1_queue.py'),'receipts','--root',t,'--data',t],capture_output=True,text=True)
            r=json.loads(p.stdout); self.assertTrue(r['SMOKE'].startswith('没通过')); self.assertTrue(r['COMPATIBLE'].startswith('没通过'))
class V55(unittest.TestCase):
    """v5.5：ModelScope 出错时改从 HF_ENDPOINT 下载，并逐文件核对（用假的 huggingface_hub，不联网）。"""
    def _setup(self,t,tamper=None,missing=None,sha_ret='c'*40,ms_ok=False):
        import types, hashlib, importlib.util
        spec=importlib.util.spec_from_file_location('up_helpers',HERE/'upstream_qwen.py')
        src=(HERE/'upstream_qwen.py').read_text()
        ns={}; exec(src[src.index('def file_digest'):src.index('MODEL_SKIP_CHECK')]+'MODEL_SKIP_CHECK = {".gitattributes", "README.md", "LICENSE"}',{'hashlib':hashlib},ns)
        snap=Path(t)/'snap'; snap.mkdir()
        files={'model.safetensors':b'weights'*100,'config.json':b'{"a":1}','README.md':b'readme'}
        for n,b in files.items(): (snap/n).write_bytes(b)
        sibs=[]
        for n,b in files.items():
            if n.endswith('.safetensors'): sibs.append(types.SimpleNamespace(rfilename=n,lfs={'sha256':hashlib.sha256(b).hexdigest()},blob_id=None))
            else: sibs.append(types.SimpleNamespace(rfilename=n,lfs=None,blob_id=hashlib.sha1(b'blob %d\0'%len(b)+b).hexdigest()))
        if tamper: (snap/tamper).write_bytes(b'tampered')
        if missing: (snap/missing).unlink()
        calls=[]
        hub=types.ModuleType('huggingface_hub')
        class Api:
            def model_info(self,model,revision,files_metadata): calls.append(('info',model,revision)); return types.SimpleNamespace(sha=sha_ret,siblings=sibs)
        hub.HfApi=Api; hub.snapshot_download=lambda model,revision: calls.append(('snap',model,revision)) or str(snap)
        def ms(rev):
            if ms_ok: return ('ms_path',rev,'modelscope')
            raise RuntimeError('[500] 获取模型目录树失败')
        q=types.SimpleNamespace(MODEL='Qwen/Qwen3-8B',fetch_from_modelscope=ms,file_digest=ns['file_digest'],MODEL_SKIP_CHECK=ns['MODEL_SKIP_CHECK'])
        return q,hub,calls,snap
    def _run(self,q,hub,rev='c'*40):
        import worker
        old=sys.modules.get('huggingface_hub'); sys.modules['huggingface_hub']=hub; os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
        try: return worker.fetch_verified(q,rev)
        finally:
            if old is None: sys.modules.pop('huggingface_hub',None)
            else: sys.modules['huggingface_hub']=old
    def test_modelscope_ok_skips_hf(self):
        with tempfile.TemporaryDirectory() as t:
            q,hub,calls,_=self._setup(t,ms_ok=True); self.assertEqual(self._run(q,hub)[0],'ms_path'); self.assertEqual(calls,[])
    def test_fallback_verified(self):
        with tempfile.TemporaryDirectory() as t:
            q,hub,calls,snap=self._setup(t,tamper='README.md')   # README 在不核对的名单里
            path,rev,src=self._run(q,hub); self.assertEqual((path,rev),(str(snap),'c'*40)); self.assertIn('逐文件核对',src)
            self.assertEqual(calls,[('info','Qwen/Qwen3-8B','c'*40),('snap','Qwen/Qwen3-8B','c'*40)])
    def test_fallback_rejects_bad_files(self):
        for kw in ({'tamper':'model.safetensors'},{'tamper':'config.json'},{'missing':'config.json'},{'sha_ret':'d'*40}):
            with tempfile.TemporaryDirectory() as t:
                q,hub,_,_=self._setup(t,**kw)
                with self.assertRaises(ValueError,msg=str(kw)): self._run(q,hub)
    def test_download_timeout_two_hours(self):
        src=(HERE/'a1_queue.py').read_text(); self.assertEqual(src.count("timeout=7200 if kind=='download'"),2); self.assertNotIn('1200',src)
    def test_both_downloads_use_fallback(self):
        for n in ('worker.py','compat.py'):
            s=(HERE/n).read_text(); self.assertNotIn('q.fetch_from_modelscope(',s.replace('try: return q.fetch_from_modelscope(rev)',''))
if __name__=='__main__': unittest.main()
