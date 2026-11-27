"""
Checks for the single-instruction steering (rosie.instance_edit.edit_single, infer_single.py). CPU, tokenizer files
only.

    python check_single.py [--sample_ids 06 09 276]
        1. the real tokenizer and VL processor with a stub encoder (check_binding.check_tokens'): the pipeline's own
           encode_prompt on edit_inst_single gives the layout's text length, the slot holds the instruction's tokens,
           and each found phrase's positions decode to the phrase; then the found rate and the longest instruction
           over both benches
        2. the pipeline's own __call__ (check_binding.tiny_pipeline): no steering, and steering at strengths 0, ==
           the baseline's call (same pixels); steering on: one stats entry per step and block, the instruction term
           acting
"""
import argparse
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

OK = True


def report(name, cond, extra=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {extra}")


def runs(pos):
    """Split sorted positions into contiguous runs."""
    out = []
    for p in pos:
        if out and p == out[-1][-1] + 1:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def check_tokens(args):
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    from bench import Bench
    from longcat import MODEL_ID, LongCatImageEditPipeline
    from rosie.instance_edit import prepare_single
    from rosie.tokens import Prompter

    tok = Qwen2Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    proc = Qwen2VLProcessor.from_pretrained(MODEL_ID, subfolder="text_processor")
    pipe = LongCatImageEditPipeline(scheduler=None, vae=None, text_encoder=None, tokenizer=tok,
                                    text_processor=proc, transformer=None)

    def stub(input_ids, attention_mask, pixel_values, image_grid_thw, output_hidden_states):
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32)
        return SimpleNamespace(hidden_states=[torch.stack([pos, input_ids[0].float()], -1)[None]])

    pipe.text_encoder = stub
    P = Prompter.from_pipe(pipe)
    bench = Bench(args.dataset_root)
    for sid in args.sample_ids:
        s = bench[sid]
        prep = prepare_single(P, s)
        L = prep["layout"]
        emb, txt_ids = pipe.encode_prompt(prompt=prep["content"], image=P.prompt_image(s["image"], prep["w"], prep["h"]))
        ids = emb[0, :, 1].long()
        print(f"{sid} {L.summary()} found {sum(prep['found'])}/{len(prep['found'])}")
        report("   the pipeline's text stream has the layout's length, its positions on the diagonal",
               emb.shape[1] == L.n_text and torch.equal(txt_ids, L.text_ids()))
        report("   the slot holds the instruction's tokens", ids[L.n_vis:L.n_vis + len(prep["ids"])].tolist() == prep["ids"])
        bad = [(ph, tok.decode(ids[r].tolist())) for ph, w in zip(prep["phrases"], L.tgt_words)
               for r in runs(w.tolist())
               if not re.fullmatch(re.escape(ph) + r"(?:s|es)?", tok.decode(ids[r].tolist()).strip(), re.IGNORECASE)]
        report("   every occurrence of a found phrase decodes to the phrase (or its plural)", not bad, f"{bad[:3]}")

    for root in (args.dataset_root, "../3D-bench/image/lomoe_bench"):
        b = Bench(root)
        found, longest, miss = [], 0, []
        for sid in b.ids:
            if not b.paired(sid):
                continue
            prep = prepare_single(P, b[sid])
            found += prep["found"]
            longest = max(longest, len(prep["ids"]))
            miss += [(sid, ph) for ph, f in zip(prep["phrases"], prep["found"]) if not f]
        report(f"{root}: every instruction fits the slot", longest <= P.slot_len,
               f"(longest {longest} of {P.slot_len}); phrases found {sum(found)}/{len(found)} "
               f"({100 * sum(found) / len(found):.0f}%), e.g. missing {miss[:4]}")


def check_pipeline(args):
    from bench import Bench
    from check_binding import tiny_pipeline
    from rosie.instance_edit import edit_single
    from rosie.steer import RotationSteer

    pipe = tiny_pipeline()
    s = Bench(args.dataset_root)[args.sample_ids[0]]
    steps = 3
    eq = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b))  # noqa: E731
    with torch.no_grad():
        plain = pipe(image=s["image"], prompt=s["edit_inst_single"], negative_prompt="", num_inference_steps=steps,
                     guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        none, _ = edit_single(pipe, s, steps=steps)
        zero, _ = edit_single(pipe, s, steps=steps, steer=RotationSteer(rot_ctc=0, rot_src=0, rot_bg=0, rot_ins=0))
        report("no steering == the baseline's call (pixels)", eq(none, plain))
        report("strengths 0 == the baseline's call (pixels)", eq(zero, plain))
        steer = RotationSteer(ins_centre="prompts")
        out, prep = edit_single(pipe, s, steps=steps, steer=steer)
        n_blocks = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
        ins = [e["ins_deg"] for e in steer.stats]
        report(f"steering on: {len(steer.stats)} stats entries = {steps} steps x {n_blocks} blocks, image changed, "
               f"instruction term acting", len(steer.stats) == steps * n_blocks and not eq(out, plain) and
               max(ins) > 0, f"(found {sum(prep['found'])}/{len(prep['found'])}, ins_deg max {max(ins):.1f})")
        report("the hooks are gone afterwards", not any(b._forward_hooks for b in pipe.transformer.transformer_blocks)
               and not pipe.transformer._forward_pre_hooks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_ids", type=str, nargs="+", default=["06", "09", "276"])
    args = p.parse_args()
    check_tokens(args)
    check_pipeline(args)
    print("PASS" if OK else "FAIL")
    sys.exit(0 if OK else 1)
