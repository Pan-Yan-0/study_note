"""Build train_augb_v6.py from train_augb_v5.py (v5.2) by exact, once-only replacements (Commit 5: B2 + NLIinit against B2)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)


rep('''"""DiCo-NLI AugB v5.2 (Claude, 11-12;''',
    '''"""DiCo-NLI AugB v6 (Claude, 11-12): v5.2 + Commit 5 = B2-NLIinit (the +2 data of B2-Haiku, NLI-checkpoint initialisation)
against B2-Haiku rerun in the same queue (no B2 SMOKE was archived in Commit 2, so B2 is not reused). B2-NLIinit has the same
rows, order and update schedule as B2-Haiku; the only change is the NLIinit arm of Commit 4. Commit 5 keeps the v5.1 software
checks (workers and restored tasks must match the SMOKE run's torch / cuda / transformers), and its formal queue also requires
the SMOKE B2-NLIinit to have loaded the pinned NLI encoder exactly as in Commit 4 (nli_smoke_problems).

v5.2 docstring follows.
DiCo-NLI AugB v5.2 (Claude, 11-12;''')

rep('''              "BiBCE": {"kind": "orig", "aux": "bibce_mean_of_two_directions", "bibce_lambda": 0.25}}
GROUPS = tuple(GROUP_SPEC)''',
    '''              "BiBCE": {"kind": "orig", "aux": "bibce_mean_of_two_directions", "bibce_lambda": 0.25},
              # v6 (Commit 5): B2-Haiku's rows with the NLIinit arm
              "B2-NLIinit": {"kind": "gen", "k": 2, "init": "nli", "init_model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                             "init_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "init_prefixes": ["deberta.", "pooler."]}}
GROUPS = tuple(GROUP_SPEC)''')

rep('''PAIRED = [("B2-Haiku", "step+2"), ("B4-Haiku", "step+4")]''',
    '''PAIRED = [("B2-Haiku", "step+2"), ("B4-Haiku", "step+4"), ("B2-NLIinit", "B2-Haiku")]''')

rep('''           "4": ["OLat4", "NLIinit", "BiBCE"]}  # v5: same rule''',
    '''           "4": ["OLat4", "NLIinit", "BiBCE"],  # v5: same rule
           "5": ["B2-Haiku", "B2-NLIinit"]}  # v6: B2 rerun, no reuse''')

rep('''                "4": ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"]}''',
    '''                "4": ["B0", "OLat1", "OLat4", "NLIinit", "BiBCE"], "5": ["B2-Haiku", "B2-NLIinit"]}''')

rep('''          "code_version": "v5.2 (Claude): v4.1 + OLat / NLIinit / BiBCE arms; Commit 4 software identity, receipt binding and complete receipt values",''',
    '''          "code_version": "v6 (Claude): v5.2 + Commit 5 (B2-NLIinit against B2-Haiku)",''')

rep('''    if commit == "2":
        sd = d / "scale"''', '''    if commit in ("2", "5"):  # v6: Commit 5 needs the +2 data too
        sd = d / "scale"''')

rep('''        if RUN_COMMIT and RUN_COMMIT[0] == "4" and soft_env(d.get("environment")) != soft_env(envinfo()):''',
    '''        if RUN_COMMIT and RUN_COMMIT[0] in ("4", "5") and soft_env(d.get("environment")) != soft_env(envinfo()):''')

rep('''    parser.add_argument("--commit",choices=["1","2","3","4"],''', '''    parser.add_argument("--commit",choices=["1","2","3","4","5"],''')

rep('''    if args.commit == "4":
        # v5.1: worker software = SMOKE software; receipt complete and bound to this SMOKE run; OLat1 = B0 recomputed read-only;
        # restored tasks must come from the same software. All of this happens before the first task.
        drift = sorted(g for g in SMOKE_GROUPS["4"] if soft_env(jload(out/"smoke"/g/"done.json").get("environment")) != soft_env(info))
        if drift:
            die("torch / cuda / transformers of this worker %s differ from the SMOKE run (%s): rerun --stage smoke --group all "
                "in this environment first" % (soft_env(info), drift))
        probs = arm_receipt_problems(out, rev, info)
        if probs:
            die("arm_checks.json does not validate %s: rerun --stage smoke --group all" % probs)
        if olat1_equivalence(out, write=False)["status"] != "pass":
            die("OLat1 = B0 check fails on the SMOKE artifacts: formal queue blocked")
        stale = env_mismatched_tasks(out, ["B0"] + COMMITS["4"], soft_env(info))''',
    '''    if args.commit in ("4", "5"):
        # v5.1: worker software = SMOKE software; receipt complete and bound to this SMOKE run; OLat1 = B0 recomputed read-only;
        # restored tasks must come from the same software. All of this happens before the first task.
        # v6: Commit 5 keeps the software checks; the arm receipt and OLat1 = B0 belong to Commit 4 only.
        drift = sorted(g for g in SMOKE_GROUPS[args.commit] if soft_env(jload(out/"smoke"/g.replace("+", "plus")/"done.json").get("environment")) != soft_env(info))
        if drift:
            die("torch / cuda / transformers of this worker %s differ from the SMOKE run (%s): rerun --stage smoke --group all "
                "in this environment first" % (soft_env(info), drift))
        if args.commit == "4":
            probs = arm_receipt_problems(out, rev, info)
            if probs:
                die("arm_checks.json does not validate %s: rerun --stage smoke --group all" % probs)
            if olat1_equivalence(out, write=False)["status"] != "pass":
                die("OLat1 = B0 check fails on the SMOKE artifacts: formal queue blocked")
        if args.commit == "5":
            probs = nli_smoke_problems(out)
            if probs:
                die("Commit 5 SMOKE init record does not validate %s: rerun --stage smoke --group all" % probs)
        stale =env_mismatched_tasks(out, (["B0"] if args.commit == "4" else []) + COMMITS[args.commit], soft_env(info))''')

rep('''B0_REUSE_IGNORE_TASK = (''', '''def nli_smoke_problems(out):
    """v6, Commit 5: the SMOKE B2-NLIinit must have loaded the pinned NLI encoder (200 tensors, 278,809,344 parameters,
    only the B0 classifier kept, no forward-config difference, global RNG restored); B2-Haiku must have no arm."""
    probs = []
    a = jload(Path(out) / "smoke" / "B2-Haiku" / "smoke.json").get("arm_report")
    if a is not None:
        probs.append("B2-Haiku arm_report")
    r = jload(Path(out) / "smoke" / "B2-NLIinit" / "smoke.json").get("arm_report") or {}
    n, spec = r.get("nli_init") or {}, GROUP_SPEC["B2-NLIinit"]
    if not (r.get("global_rng_restored") is True and n.get("model") == spec["init_model"] and n.get("revision") == spec["init_revision"]
            and n.get("tensors_loaded") == 200 and n.get("parameters_loaded") == 278809344
            and n.get("kept_from_b0") == ["classifier.bias", "classifier.weight"] and n.get("forward_config_differences_kept_b0") == {}):
        probs.append("B2-NLIinit nli_init")
    return probs


B0_REUSE_IGNORE_TASK = (''')

open(dst, "w", encoding="utf-8").write(s)
print("ok", dst)
