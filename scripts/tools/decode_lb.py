"""Decode a leaderboard result of a candidate relative to the final submission (LB 0.9357) into how many of the changed
rows were right. The leaderboard shows 4 decimals (binary F1, macro F1) only when a team's best improves; otherwise the
submission history shows the total with 2 decimals (coarse mode: every solution inside that rounding interval).

The constants below are the per-class state of the final submission on the leaderboard's answer key, solved from its
leaderboard numbers (binary F1 0.9579 = 2*182/(185+195), macro F1 0.9134). The leaderboard key's class totals are
적절 405, 무의미형 5, 파일명형 8, 중복형 35, 장식오용형 6, 불충분형 132, 무관형 9 (195 부적절); see README "리더보드 채점 정답".

usage: python scripts/tools/decode_lb.py <candidate.csv> <binary F1> <macro F1>     (4-decimal values)
       python scripts/tools/decode_lb.py <candidate.csv> <2-decimal total score>    (coarse)
For every group of changed rows (from b -> to a) it enumerates: x rows truly b, y rows truly a, the rest neither
(then truly 적절 or not, which matters for the binary half only)."""
import collections
import csv
import itertools
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "submission", "final_submission_0.9357.csv")
LAB = ["적절", "무의미형", "파일명형", "중복형", "장식오용형", "불충분형", "무관형"]
TP = {"적절": 402, "무의미형": 5, "파일명형": 8, "중복형": 33, "장식오용형": 5, "불충분형": 123, "무관형": 4}
TRUE = {"적절": 405, "무의미형": 5, "파일명형": 8, "중복형": 35, "장식오용형": 6, "불충분형": 132, "무관형": 9}
BIN_TP, BIN_TRUE = 182, 195

rd = lambda p: {r["record_id"]: r["label"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
base, cand = rd(BASE), rd(sys.argv[1])
coarse = len(sys.argv) == 3
obs_b, obs_m = (float(sys.argv[2]), None) if coarse else (float(sys.argv[2]), float(sys.argv[3]))
pred = collections.Counter(cand.values())
groups = collections.Counter((base[k], cand[k]) for k in base if base[k] != cand[k])
print("changes:", dict(groups))
opts = []
for (b, a), n in groups.items():
    o = []
    for x in range(n + 1):
        for y in range(n + 1 - x):
            rest = n - x - y
            for z in (range(rest + 1) if "적절" not in (b, a) else [0]):
                o.append((b, a, x, y, rest, z))
    opts.append(o)
sols = []
for combo in itertools.product(*opts):
    tp, btp = dict(TP), BIN_TP
    for b, a, x, y, rest, z in combo:
        tp[b] -= x
        tp[a] += y
        tb, ta = b != "적절", a != "적절"
        npos = x * tb + y * ta + (rest - z)
        if not tb and ta:
            btp += npos
        elif tb and not ta:
            btp -= npos
    f1 = [2 * tp[c] / (pred[c] + TRUE[c]) if pred[c] + TRUE[c] else 0 for c in LAB]
    m = sum(f1) / 7
    bf = 2 * btp / (sum(v for c, v in pred.items() if c != "적절") + BIN_TRUE)
    ok = (abs(round((bf + m) / 2, 2) - obs_b) < 1e-9) if coarse else (abs(round(bf, 4) - obs_b) < 1e-9 and abs(round(m, 4) - obs_m) < 1e-9)
    if ok:
        sols.append((combo, bf, m))
print(f"{len(sols)} consistent solution(s)")
sols.sort(key=lambda t: -(t[1] + t[2]))
if coarse and sols:
    print(f"  score range {min((b + m) / 2 for _, b, m in sols):.5f} .. {max((b + m) / 2 for _, b, m in sols):.5f}")
for combo, bf, m in sols[:20]:
    print(f"  bin {bf:.5f} macro {m:.5f} score {(bf + m) / 2:.5f} | " +
          "; ".join(f"{b}->{a}: {x} truly {b}, {y} truly {a}, {r} other ({z} 적절)" for b, a, x, y, r, z in combo))
