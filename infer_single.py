"""
RoSIE without binding on LongCat-Image-Edit-Turbo: the official one-instruction edit (infer_baseline.py's call,
edit_inst_single, guidance 1) with the rotation steering of rosie/steer.py on top. No instance prompts and no
attention restriction: the masks are used only by the steering, and instance k's instruction anchor is the update of
its target phrase (fg_prompt without the article) inside the instruction. Instances whose phrase is not in the
instruction get no instruction term; the other terms need only the mask.

Steering flags and defaults are infer_steer.py's (mice's final strengths); run_all.sh passes the tuned ones. With
every strength 0 the result is the baseline's (same call, same seed). {id}_layout.json has the token counts and
which phrases were found, {id}_steer.json the per-(step, block) turns and cosines.

    python infer_single.py --sample_ids 06 09 276 --exp_name single_smoke --cpu_offload
    python infer_single.py --sample_ids $(tr , ' ' < ../mice/paper/ablation_ids.txt) --exp_name single_50 \
        --rot_ctc 0.5 --rot_src 0.6 --rot_bg 0.35 --rot_ins 0.025 --ins_centre prompts --cpu_offload
"""
import argparse
import json
import time

import torch
from tqdm import tqdm

from bench import Bench
from infer_steer import add_steer_args, check_model, make_steer
from longcat import MODEL_ID, load_pipeline
from rosie.instance_edit import edit_single
from runs import add_run_args, run_folder, save_environment, save_result, todo


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID, help="guidance 1 only, so the Turbo model")
    p.add_argument("--prompt_key", type=str, default="edit_inst_single")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="default: the model card's (Turbo 8)")
    p.add_argument("--no_strict", action="store_true", help="keep overlapping masks (mice: --strict_masks on)")
    add_steer_args(p)
    args = p.parse_args()
    check_model(p, args)
    return args, make_steer(p, args)


def main():
    args, steer = parse_args()
    save_dir = run_folder(args, "longcatsingle")
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

    secs, found = [], []
    for sid in tqdm(ids):
        s = bench[sid]
        t0 = time.perf_counter()
        result, prep = edit_single(pipe, s, strict=not args.no_strict, seed=args.seed, steps=args.steps, steer=steer,
                                   prompt_key=args.prompt_key)
        secs.append(time.perf_counter() - t0)
        found += prep["found"]
        (save_dir / f"{sid}_layout.json").write_text(json.dumps(
            dict(prep["layout"].summary(), instruction_tokens=len(prep["ids"]), phrases=prep["phrases"],
                 found=prep["found"])))
        (save_dir / f"{sid}_steer.json").write_text(json.dumps(steer.stats))
        save_result(result, s["image"], save_dir / f"{sid}_result.png")

    stats = dict(images=len(secs), sec_per_image=sum(secs) / len(secs), first_image_sec=secs[0],
                 peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                 phrases_found=f"{sum(found)}/{len(found)}")
    (save_dir / f"stats_shard{args.shard_id}.json").write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
