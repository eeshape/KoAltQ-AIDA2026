"""Cell dispatcher: `python -m koaltq2.run <cell> [args]` (each notebook cell runs in its own process)."""

from __future__ import annotations

import importlib
import sys
import time
import traceback

CELLS = {
    "x0": "koaltq2.x0_setup",
    "x1": "koaltq2.x1_gbm",
    "x2": "koaltq2.x2_ocr",
    "x3": "koaltq2.x3_frozen",
    "x6": "koaltq2.x6_lora",
    "x7": "koaltq2.x7_stack",
    "x12": "koaltq2.x12_attr",
    "x8": "koaltq2.x8_blend",
    "lb": "koaltq2.lbest",
    "x14": "koaltq2.x14_judge",
    "x15": "koaltq2.x15_judge_lr",
    "x17": "koaltq2.x17_boundary",
    "harness": "koaltq2.harness",
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in CELLS:
        print("usage: python -m koaltq2.run {" + ",".join(CELLS) + "} [args]")
        return 2
    cell, rest = argv[0], argv[1:]
    t0 = time.time()
    print(f"[run] {cell} start {time.strftime('%H:%M:%S')} args={rest}", flush=True)
    try:
        mod = importlib.import_module(CELLS[cell])
        if rest and hasattr(mod, "cli"):
            mod.cli(rest)
        else:
            mod.main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"[run] {cell} FAILED after {time.time() - t0:.1f}s", flush=True)
        return 1
    print(f"[run] {cell} DONE in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
