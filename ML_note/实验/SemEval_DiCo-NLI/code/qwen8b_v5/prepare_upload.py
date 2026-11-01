"""Build one offline upload ZIP on the user's computer, without model weights."""
import argparse, zipfile
from pathlib import Path
from common import HERE, require
p=argparse.ArgumentParser(); p.add_argument('--official-zip',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True); p.add_argument('--baseline-cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
require(not a.output.exists(),'输出已存在，请换文件名')
require((a.adapter/'adapter_model.safetensors').is_file(),'4B adapter缺失')
require(a.baseline_cache.is_file(),'历史零样本缓存缺失')
with zipfile.ZipFile(a.official_zip) as z:
    require(any(n.endswith('final_data/train/dico_nli_train_track1_participant_labeled.csv') for n in z.namelist()),'不是官方数据包')
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(HERE.iterdir()):
        if f.is_file(): z.write(f,'qwen8b_v5/'+f.name)
    z.write(a.official_zip,'official.zip'); z.write(a.baseline_cache,'inputs/zero4b_cache.jsonl')
    for f in sorted(a.adapter.iterdir()):
        if f.is_file() and f.name in ('adapter_model.safetensors','adapter_config.json'): z.write(f,'inputs/adapter4b/'+f.name)
print(a.output)
