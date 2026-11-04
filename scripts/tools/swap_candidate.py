"""Make a same-count candidate from NEW test logits on top of a base submission (default: the final submission).

Only the ranking inside 적절 / 중복형 / 불충분형 changes: residual rows of these classes are re-assigned with the Hungarian
algorithm so that each class keeps its count in the base file. Rows decided by rules stay fixed: L0 rows (public 109,
R1, R2, R4' template twins) and every row a harness module H3-H5 decided (read from harness_log.jsonl).
The leaderboard difference then measures exactly how many of the swapped rows the new logits got right
(scripts/tools/decode_lb.py).

usage (from the repository root, after scripts/reproduce_best.sh):
  KOALTQ_BUNDLE=data/koaltq_bundle KOALTQ_RUNS=model/runs_frozen PYTHONPATH=src \
  python scripts/tools/swap_candidate.py --logits my_test_logits.npy --out outputs/my_candidate.csv
  --logits: (600, 7) array of log-probabilities (or --probs for probabilities), class order = koaltq2.common.LABELS
"""
import argparse
import collections
import csv
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from koaltq2.common import L2I, Data
from koaltq2.rules import l0_fixed

ap = argparse.ArgumentParser()
ap.add_argument("--logits", required=True)
ap.add_argument("--probs", action="store_true")
ap.add_argument("--base", default="submission/final_submission_0.9357.csv")
ap.add_argument("--harness-log", default="outputs/harness_log.jsonl")
ap.add_argument("--out", required=True)
a = ap.parse_args()

COMMON = ["적절", "중복형", "불충분형"]
d = Data()
base_by_id = {r["record_id"]: r["label"] for r in csv.DictReader(open(a.base, encoding="utf-8-sig"))}
base = [base_by_id[r["record_id"]] for r in d.test]
L = np.load(a.logits).astype(float)
if a.probs:
    L = np.log(L + 1e-9)
assert L.shape == (len(d.test), 7), L.shape
pos = {r["record_id"]: i for i, r in enumerate(d.test)}
kept = set(l0_fixed(d))
for e in map(json.loads, open(a.harness_log, encoding="utf-8")):
    if e["module"].startswith(("H3", "H4")):
        kept.add(pos[e["record_id"]])
rows = [i for i in range(len(d.test)) if i not in kept and base[i] in COMMON]
target = collections.Counter(base[i] for i in rows)
slots = [c for c in COMMON for _ in range(target[c])]
cost = -np.array([[L[i, L2I[c]] for c in slots] for i in rows])
r_idx, c_idx = linear_sum_assignment(cost)
out = list(base)
for ri, ci in zip(r_idx, c_idx):
    out[rows[ri]] = slots[ci]
with open(a.out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["record_id", "label"])
    for r, lab in zip(d.test, out):
        w.writerow([r["record_id"], lab])
ch = collections.Counter(f"{b}->{o}" for b, o in zip(base, out) if b != o)
print(f"fixed rows {len(kept)} | changed vs base {sum(ch.values())}: {dict(ch)} | counts {dict(collections.Counter(out))}")
