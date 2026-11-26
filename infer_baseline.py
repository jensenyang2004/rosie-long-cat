"""
Baseline: the official LongCat-Image-Edit-Turbo edit (vendored pipeline, no instance control) on MICE-Bench /
LoMOE-Bench, one instruction per image (edit_inst_single). Steps and guidance default to the model card's (Turbo:
8 steps, guidance 1). Run folder conventions in runs.py; stats_shard<k>.json has the time per image and peak GPU
memory. On 48 GB cards add --cpu_offload (without it the second image ran out of memory).

    python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --sample_ids 06 09 --exp_name smoke
    ./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice
    ./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/lomoe_bench --exp_name turbo_lomoe
    python infer_baseline.py --model meituan-longcat/LongCat-Image-Edit --exp_name base_mice    (50 steps, CFG 4.5)
"""
import argparse
import json
import time

import torch
from tqdm import tqdm

from bench import Bench
from longcat import MODEL_ID, SAMPLING, load_pipeline
from runs import add_run_args, run_folder, save_environment, save_result, todo


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID)
    p.add_argument("--prompt_key", type=str, default="edit_inst_single")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="default: the model card's (Turbo 8, base 50)")
    p.add_argument("--guidance", type=float, default=None, help="guidance_scale; default: the model card's "
                   "(Turbo 1 = no CFG, base 4.5)")
    args = p.parse_args()
    if args.model in SAMPLING:  # record the values used in args.json, not None
        steps, guidance = SAMPLING[args.model]
        args.steps = steps if args.steps is None else args.steps
        args.guidance = guidance if args.guidance is None else args.guidance
    elif args.steps is None or args.guidance is None:
        p.error(f"--model {args.model}: pass --steps and --guidance")
    return args


def main():
    args = parse_args()
    save_dir = run_folder(args, "longcat")
    bench = Bench(args.dataset_root)
    ids = todo(bench, args, save_dir)
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
        result = pipe(
            image=s["image"],
            prompt=s[args.prompt_key],
            negative_prompt="",  # as in the model card; unused at guidance 1
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        secs.append(time.perf_counter() - t0)
        save_result(result, s["image"], save_dir / f"{sid}_result.png")

    stats = dict(images=len(secs), sec_per_image=sum(secs) / len(secs), first_image_sec=secs[0],
                 peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None)
    (save_dir / f"stats_shard{args.shard_id}.json").write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
