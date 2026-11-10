"""在本机生成唯一一个上传包（不含模型底座）。

    python qwen4b_augb/prepare_upload.py --official-zip <官方仓库 zip> --adapter <4B adapter 目录> --output a10_upload_qwen4b_augb.zip

只打包白名单里的文件（common.SHIP、随包的零样本缓存和实验 B 的数据），输出先写临时文件、成功后再改名，
输出路径不能和任何输入是同一个文件，所以输出放在代码目录里也不会把自己打进去。
不给 --adapter 时只生成"缺 adapter"的包，并在 inputs/adapter4b/ 放一个说明；A10 上 prepare 会因此停下。
"""
import argparse, os, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, require, ship_files, sha

ADAPTER_FILES = ('adapter_model.safetensors', 'adapter_config.json')


def build(official_zip, adapter, output):
    official_zip = Path(official_zip).resolve(); output = Path(output).resolve()
    require(not output.exists(), '输出已存在，请换文件名：%s' % output)
    with zipfile.ZipFile(official_zip) as z:
        require(any(n.endswith('final_data/train/dico_nli_train_track1_participant_labeled.csv') for n in z.namelist()), '不是官方数据包')
    entries = [(HERE / n, 'qwen4b_augb/' + n) for n in ship_files()]          # 先冻结输入列表，再创建输出
    entries.append((HERE / 'baseline' / 'zero4b_cache_a10_t1.jsonl', 'inputs/zero4b_cache.jsonl'))
    entries.append((official_zip, 'official.zip'))
    if adapter is not None:
        adapter = Path(adapter).resolve()
        for name in ADAPTER_FILES:
            require((adapter / name).is_file(), '4B adapter 目录里缺 %s' % name)
            entries.append((adapter / name, 'inputs/adapter4b/' + name))
    for src, _ in entries:
        require(src.resolve() != output, '输出不能和输入是同一个文件：%s' % src)
    tmp = output.with_name(output.name + '.tmp')
    require(not tmp.exists(), '临时文件已存在：%s' % tmp)
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for src, arc in entries:
                z.write(src, arc)
            if adapter is None:
                z.writestr('inputs/adapter4b/把两个文件放在这里.txt',
                           '这个包还缺 4B LoRA：把 adapter_model.safetensors 和 adapter_config.json 放进这个文件夹（同名放在 inputs/adapter4b/ 下）。\n')
            z.writestr('UPLOAD_MANIFEST.json', __import__('json').dumps(
                {arc: sha(src) for src, arc in entries}, ensure_ascii=False, indent=1))
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--official-zip', type=Path, required=True)
    p.add_argument('--adapter', type=Path, help='4B LoRA 目录（含 adapter_model.safetensors、adapter_config.json）；不给就生成缺 adapter 的包')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(build(a.official_zip, a.adapter, a.output))
    if a.adapter is None:
        print('注意：这个包里还没有 4B adapter。')
