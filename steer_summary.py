"""
Summary of a steering run's {id}_steer.json: per step and block kind (double / single), the mean turn of the
instance tokens, of each term, the cosine of the update to the instruction anchor (sem_cos_inst, averaged over the
instances), and the fraction of tokens at max_turn. --mice puts mice's run of the same sample next to it (klein has
4 steps, 5 double + 20 single blocks; same keys).

    python steer_summary.py results/longcatrot_rot_smoke 09
    python steer_summary.py results/longcatrot_rot_smoke 09 --mice ../3D-bench/baselines/4B_ICRS_rotation_mice
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

KEYS = ("turn_inst", "turn_bg", "ins_deg", "src_deg", "ctc_deg", "bg_deg", "sem_cos", "ins_raw_cos", "capped")


def rows(path: Path) -> dict:
    """(step, kind) -> {key: mean over the blocks of that kind}."""
    acc = defaultdict(lambda: defaultdict(list))
    for e in json.loads(path.read_text()):
        kind = e["layer"].split("_")[0]
        cos = [c for c in e.get("sem_cos_inst", []) if c is not None and not math.isnan(c)]
        vals = dict(e, sem_cos=sum(cos) / len(cos) if cos else None)
        for k in KEYS:
            if vals.get(k) is not None:
                acc[(e["step"], kind)][k].append(vals[k])
        acc[(e["step"], kind)]["sigma"].append(e["sigma"])
    return {key: {k: sum(v) / len(v) for k, v in d.items()} for key, d in sorted(acc.items())}


def show(name: str, table: dict):
    print(f"{name}\n  step kind     sigma " + " ".join(f"{k:>9s}" for k in KEYS))
    for (step, kind), d in table.items():
        print(f"  {step:4d} {kind:7s} {d['sigma']:7.3f} " +
              " ".join(f"{d[k]:9.3f}" if k in d else f"{'-':>9s}" for k in KEYS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("sample_id", type=str)
    p.add_argument("--mice", type=Path, default=None, help="a mice run folder with {id}_steer.json")
    args = p.parse_args()
    show(f"{args.run} / {args.sample_id} (degrees; capped = fraction of tokens)",
         rows(args.run / f"{args.sample_id}_steer.json"))
    if args.mice:
        show(f"\nmice {args.mice.name} / {args.sample_id}", rows(args.mice / f"{args.sample_id}_steer.json"))


if __name__ == "__main__":
    main()
