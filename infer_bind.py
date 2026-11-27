"""
Text binding only (no steering): mice's final binding on LongCat-Image-Edit-Turbo, global prompt " " plus one
"Replace the {src} with {tgt}" prompt per instance (rosie/). Run folder conventions in runs.py; {id}_layout.json
records the token counts of each sample and vision_diff (should be 0); stats_shard<k>.json the time per image and
peak GPU memory. On 48 GB cards add --cpu_offload.

    python infer_bind.py --sample_ids 06 09 276 --exp_name bind_smoke --cpu_offload
    python infer_bind.py --sample_ids 06 09 276 --exp_name bind_smoke_stacked --prompt_pos stacked --cpu_offload
    python infer_bind.py --sample_ids 06 --exp_name nobind --no_bind --cpu_offload   (same prompts, stacked, the
                                                                                      model's own attention)
    ./run_2gpu.sh infer_bind.py --exp_name bind_mice --cpu_offload
"""
import argparse
import json
import time

import torch
from tqdm import tqdm

from bench import Bench
from longcat import MODEL_ID, SAMPLING, load_pipeline
from rosie.instance_edit import edit
from rosie.tokens import POSITIONS
from runs import add_run_args, run_folder, save_environment, save_result, todo


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID, help="guidance 1 only, so the Turbo model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="default: the model card's (Turbo 8)")
    p.add_argument("--prompt_pos", type=str, default="shared", choices=POSITIONS,
                   help="RoPE positions of the instance slots (rosie/tokens.py): shared = every slot at the global "
                        "slot's positions, the image where a one-prompt edit has it; stacked = one after another")
    p.add_argument("--no_strict", action="store_true", help="keep overlapping masks (mice: --strict_masks on)")
    p.add_argument("--no_bind", action="store_true", help="same prompts, no binding (the model's own attention); "
                   "needs --prompt_pos stacked")
    args = p.parse_args()
    if args.steps is None:
        if args.model not in SAMPLING:
            p.error(f"--model {args.model}: pass --steps")
        args.steps = SAMPLING[args.model][0]
    if args.model in SAMPLING and SAMPLING[args.model][1] != 1.0:
        p.error(f"{args.model} samples with CFG; the binding supports guidance 1 only")
    if args.no_bind and args.prompt_pos == "shared":
        p.error("--no_bind needs --prompt_pos stacked")
    return args


def main():
    args = parse_args()
    save_dir = run_folder(args, "longcatbind")
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
    cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    secs = []
    for sid in tqdm(ids):
        s = bench[sid]
        t0 = time.perf_counter()
        result, prep = edit(pipe, s, prompt_pos=args.prompt_pos, strict=not args.no_strict, bind=not args.no_bind,
                            seed=args.seed, steps=args.steps)
        secs.append(time.perf_counter() - t0)
        (save_dir / f"{sid}_layout.json").write_text(json.dumps(
            dict(prep["layout"].summary(), vision_diff=prep["vision_diff"], prompts=prep["contents"])))
        save_result(result, s["image"], save_dir / f"{sid}_result.png")

    stats = dict(images=len(secs), sec_per_image=sum(secs) / len(secs), first_image_sec=secs[0],
                 peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None)
    (save_dir / f"stats_shard{args.shard_id}.json").write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
