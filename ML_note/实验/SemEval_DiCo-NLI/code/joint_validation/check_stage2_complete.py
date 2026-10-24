"""Check saved stage2 completeness before downloading/releasing the instance."""
import json, zipfile, sys
from pathlib import Path

def check(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None: raise ValueError("ZIP CRC failed")
        names=set(z.namelist());base="dico_mdeberta_results/"
        read=lambda p: json.loads(z.read(base+p))
        choice=read("moco_m_choice.json")["m"]
        if choice != 0.999: raise ValueError("Unexpected MoCo m; verify the saved stage1 choice")
        methods=["ce","rdrop","dir_rdrop","scl",f"moco_m{choice}"]
        expected=[f"track1/{m}/seed{s}/fold{k}" for m in methods for s in range(42,47) for k in range(5)]
        missing=[p for p in expected if base+p+"/done.json" not in names]
        oof=[f"track1/{m}/seed{s}/oof" for m in methods for s in range(42,47)]
        absent=[p for p in oof if base+p+"/done.json" not in names]
        print(f"Saved stage2 tasks: {125-len(missing)}/125; OOF: {25-len(absent)}/25")
        if missing or absent:
            print("INCOMPLETE: keep running or inspect stage2.log; do not treat this as the final package.")
            return False
        for p in expected:
            d=read(p+"/done.json")
            assert d["code_sha256"]=="83bfe561f0f5ab5f9ff52b37dd9619faf34d3ebcb7a34032a14fa55f9b597f63",p
            assert d["precision"]=="bf16 autocast",p
        for p in oof:
            d=read(p+"/done.json");assert d["rows"]==3042 and set(d["scores"])=={"plain","sym"},p
            for mode in ["plain","sym"]:
                assert base+p+f"/pred_oof_{mode}.csv" in names,p
        summary=read("summary.json")
        assert summary["stage"]=="stage2" and not summary.get("oof_not_assembled"),"summary incomplete"
        assert all(p in summary["runs"] for p in expected)
        assert all(p.rsplit("/",1)[0] in summary["oof"] for p in oof)
        print("COMPLETE: 125 tasks and 25 OOF outputs present. Download this exact ZIP for independent scoring audit:")
        print(path)
        return True

if __name__=="__main__":
    if len(sys.argv)>1: path=Path(sys.argv[1])
    else:
        files=list(Path("/mnt/workspace/dico_mdeberta").glob("dico_mdeberta_results_*.zip"))
        if not files: raise SystemExit("No result ZIP found")
        path=max(files,key=lambda p:p.stat().st_mtime)
    raise SystemExit(0 if check(path) else 1)
