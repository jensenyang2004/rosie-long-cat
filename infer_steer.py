"""
RoSIE on LongCat-Image-Edit-Turbo: the text binding of infer_bind.py plus rotation steering (rosie/steer.py) of every
block's update of the target tokens. Defaults are mice's final run (4B_ROT_mice_2, the paper's Ours), every block.
Run folder conventions in runs.py; {id}_steer.json has the per-(step, block) turns and cosines, {id}_layout.json the
token counts, stats_shard<k>.json the time per image and peak GPU memory. On 48 GB cards add --cpu_offload.

    python infer_steer.py --sample_ids 06 09 276 --exp_name rot_smoke --cpu_offload
    python infer_steer.py --sample_ids 06 09 276 --exp_name rot_noins --rot_ins 0 --cpu_offload     (one term off)
    ./run_2gpu.sh infer_steer.py --exp_name rot_mice --cpu_offload
"""
import argparse
import json
import time

import torch
from tqdm import tqdm

from bench import Bench
from longcat import MODEL_ID, SAMPLING, load_pipeline
from rosie.instance_edit import edit
from rosie.steer import TERMS, RotationSteer
from rosie.tokens import POSITIONS
from runs import add_run_args, run_folder, save_environment, save_result, todo


def steps(s: str):
    """"all" -> None (every step), "none" -> (), "0,1" -> (0, 1)."""
    return None if s == "all" else () if s == "none" else tuple(int(i) for i in s.split(","))


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID, help="guidance 1 only, so the Turbo model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="default: the model card's (Turbo 8)")
    p.add_argument("--prompt_pos", type=str, default="shared", choices=POSITIONS,
                   help="RoPE positions of the instance slots (rosie/tokens.py)")
    p.add_argument("--no_strict", action="store_true", help="keep overlapping masks (mice: --strict_masks on)")
    p.add_argument("--no_bind", action="store_true", help="no text binding (the model's own attention); needs "
                   "--prompt_pos stacked")

    p.add_argument("--rot_ctc", type=float, default=0.9, help="contact: fraction of the way from each neighbour's "
                   "write to orthogonal, times the pair weight")
    p.add_argument("--rot_src", type=float, default=0.25, help="source: fraction of the way from the source write "
                   "to orthogonal (instance tokens)")
    p.add_argument("--rot_bg", type=float, default=0.35, help="background: fraction of the way to the source write")
    p.add_argument("--rot_ins", type=float, default=0.2, help="instruction: fraction of the way to the target-word "
                   "write")
    p.add_argument("--ctc_steps", type=str, default="0,1", help="steps of the contact term, \"all\" or \"none\" "
                   "(mice: 0,1 = sigma 1.0 -> 0.91; Turbo's 0,1 = 1.0 -> 0.905)")
    p.add_argument("--src_steps", type=str, default="all")
    p.add_argument("--bg_steps", type=str, default="all")
    p.add_argument("--ins_steps", type=str, default="all")
    p.add_argument("--sigma_terms", type=str, default="ctc,src,bg", help="terms scaled by the step's sigma, or none")
    p.add_argument("--blocks", type=str, default="all", help="blocks steered: \"all\", or names like "
                   "double_0,single_3 (10 double, 20 single)")
    p.add_argument("--local_tau", type=float, default=2.0, help="contact pair-weight falloff (tokens)")
    p.add_argument("--bg_margin", type=float, default=1.0, help="background tokens this close to an instance are free")
    p.add_argument("--bg_tau", type=float, default=2.0, help="background ramp width after the margin (tokens)")
    p.add_argument("--max_turn", type=float, default=90.0, help="cap on the total turn per token (deg)")
    args = p.parse_args()
    if args.steps is None:
        if args.model not in SAMPLING:
            p.error(f"--model {args.model}: pass --steps")
        args.steps = SAMPLING[args.model][0]
    if args.model in SAMPLING and SAMPLING[args.model][1] != 1.0:
        p.error(f"{args.model} samples with CFG; the binding and steering support guidance 1 only")
    if args.no_bind and args.prompt_pos == "shared":
        p.error("--no_bind needs --prompt_pos stacked")
    terms = () if args.sigma_terms == "none" else tuple(args.sigma_terms.split(","))
    if set(terms) - set(TERMS):
        p.error(f"--sigma_terms: use {','.join(TERMS)} or none")
    blocks = None if args.blocks == "all" else set(args.blocks.split(","))
    names = {f"double_{i}" for i in range(10)} | {f"single_{i}" for i in range(20)}
    if blocks is not None and blocks - names:
        p.error(f"--blocks: unknown {sorted(blocks - names)}")
    return args, terms, blocks


def main():
    args, terms, blocks = parse_args()
    save_dir = run_folder(args, "longcatrot")
    bench = Bench(args.dataset_root)
    ids = todo(bench, args, save_dir)
    unpaired = [i for i in ids if not bench.paired(i)]
    if unpaired:
        print(f"skipping {unpaired}: masks and prompts differ in number (mice has no result for them either)")
        ids = [i for i in ids if i not in set(unpaired)]
    if not ids:
        return
    pipe = load_pipeline(args.model, args.device, cpu_offload=args.cpu_offload)
    save_environment(save_dir, args, args.model)
    steer = RotationSteer(rot_ctc=args.rot_ctc, rot_src=args.rot_src, rot_bg=args.rot_bg, rot_ins=args.rot_ins,
                          ctc_steps=steps(args.ctc_steps), src_steps=steps(args.src_steps),
                          bg_steps=steps(args.bg_steps), ins_steps=steps(args.ins_steps), sigma_terms=terms,
                          blocks=blocks, local_tau=args.local_tau, bg_margin=args.bg_margin, bg_tau=args.bg_tau,
                          max_turn_deg=args.max_turn)
    cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    secs = []
    for sid in tqdm(ids):
        s = bench[sid]
        t0 = time.perf_counter()
        result, prep = edit(pipe, s, prompt_pos=args.prompt_pos, strict=not args.no_strict, bind=not args.no_bind,
                            seed=args.seed, steps=args.steps, steer=steer)
        secs.append(time.perf_counter() - t0)
        (save_dir / f"{sid}_layout.json").write_text(json.dumps(
            dict(prep["layout"].summary(), vision_diff=prep["vision_diff"], prompts=prep["contents"])))
        (save_dir / f"{sid}_steer.json").write_text(json.dumps(steer.stats))
        save_result(result, s["image"], save_dir / f"{sid}_result.png")

    stats = dict(images=len(secs), sec_per_image=sum(secs) / len(secs), first_image_sec=secs[0],
                 peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None)
    (save_dir / f"stats_shard{args.shard_id}.json").write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
