import tempfile, unittest, subprocess, sys, json
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
if __name__=='__main__': unittest.main()
