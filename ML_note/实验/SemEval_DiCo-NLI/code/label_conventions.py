"""Check two labelling conventions suggested by the blind review, on the English training set only (dev untouched).

Counts are per phrase pair (pair_id), not per row: a reversible pair has two rows, a NEGATIVE_OTHER pair has one.

1. Numbers: pairs whose two sides differ only in one number (digits or number words). Does the gold follow the
   "at least" reading (the side with the larger number entails the other) or give NEGATIVE_OTHER?
2. Structure: one side starts with a preposition and the other does not (e.g. "with a dog" / "A brown dog").
   Is NEGATIVE_OTHER more common there than in the other groups?

Usage:
    python3 label_conventions.py <official train track1 participant_labeled.csv> <out_json>
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict

TOKEN = re.compile(r"[\w,.]+")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100, "thousand": 1000, "dozen": 12}
PREPS = {"in", "on", "with", "of", "for", "at", "from", "by", "to", "near", "across", "over", "under", "into",
         "after", "before", "during", "about", "against", "behind", "between", "inside", "outside", "through"}
SHORT = {"EQUIVALENCE": "E", "FORWARD_ENTAILMENT": "F", "BACKWARD_ENTAILMENT": "B", "NEGATIVE_OTHER": "O"}


def number(tok):
    t = tok.lower()
    if re.fullmatch(r"\d+(,\d{3})*(\.\d+)?", t):
        return float(t.replace(",", ""))
    return WORDS.get(t)


def tokens(text):
    return [t.strip(".,") for t in TOKEN.findall(text) if t.strip(".,")]


def number_pair(a, b):
    """The two numbers if the sides differ in exactly one token each and both tokens are numbers."""
    ca, cb = Counter(t.lower() for t in tokens(a)), Counter(t.lower() for t in tokens(b))
    da, db = ca - cb, cb - ca
    if sum(da.values()) != 1 or sum(db.values()) != 1:
        return None
    na, nb = number(next(iter(da))), number(next(iter(db)))
    if na is None or nb is None or na == nb:
        return None
    return na, nb


def lead_prep(text):
    t = tokens(text)
    return t[0].lower() if t and t[0].lower() in PREPS else None


def main(path, out):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    nums, num_examples, num_pairs = Counter(), defaultdict(list), defaultdict(list)
    struct = defaultdict(Counter)
    for pid, rs in by_pair.items():
        r = next((x for x in rs if x["instance_id"].endswith("__original")), rs[0])
        a, b, lab = r["text1"], r["text2"], SHORT[r["label"]]
        np_ = number_pair(a, b)
        if np_:
            bigger_first = np_[0] > np_[1]
            if lab == "O":
                kind = "NEGATIVE_OTHER"
            elif lab == "E":
                kind = "EQUIVALENCE"
            elif (lab == "F") == bigger_first:
                kind = "数大的一侧推出数小的一侧"
            else:
                kind = "数小的一侧推出数大的一侧"
            nums[kind] += 1
            num_pairs[kind].append(pid)
            if len(num_examples[kind]) < 12:
                num_examples[kind].append(f"{a} | {b} → {lab}")
        pa, pb = lead_prep(a), lead_prep(b)
        if (pa is None) != (pb is None):
            group = "一侧以介词开头、另一侧不是"
        elif pa is None:
            group = "两侧都不以介词开头"
        elif pa == pb:
            group = "两侧以同一个介词开头"
        else:
            group = "两侧以不同介词开头"
        struct[group]["O" if lab == "O" else ("E" if lab == "E" else "F/B")] += 1
    result = {
        "data": path.split("/")[-1],
        "pairs": len(by_pair),
        "numbers_only_difference_pairs": dict(nums),
        "numbers_examples": dict(num_examples),
        "numbers_pair_ids": {k: sorted(v) for k, v in num_pairs.items()},
        "leading_preposition_pairs": {
            k: dict(v, NEGATIVE_OTHER_share=round(v["O"] / sum(v.values()), 3)) for k, v in struct.items()},
    }
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
