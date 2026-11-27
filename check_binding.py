"""
Checks for the token assignment and the text binding.

    python check_binding.py            CPU, no weights, a tiny random transformer on synthetic layouts:
        1. the row classes cover every row once and rebuild the rule mask (allowed_mask) exactly, also with
           overlapping instance masks
        2. bind off: BindAttnProcessor == the model's own processor (double and single blocks)
        3. bind on, one double block and one single block: changing slot 2 changes only instance 2's target and
           source tokens; changing the global slot changes the background and the source, never an instance's
           target; changing the vision span changes every target token; each slot reads only the vision span,
           itself and its instance's image tokens
        4. positions: "stacked" == the pipeline's own txt_ids and image start; "shared", one block: each instance's
           target tokens == a run with only its own prompt, and the background == a run with no instance prompt
           (with "stacked" they differ)
    python check_binding.py --tokens   the real tokenizer and VL processor (small downloads, no weights) on bench
        samples, both position modes: layout sizes; the source and target words decoded from their positions equal
        the bench's; through the encode_prompt wrapper with a stub text encoder, every kept token is the one the
        layout says (vision span, global slot, each instance slot) and the txt_ids are the layout's; the
        prepare_latents wrapper starts the image grid at the layout's img_start; the row classes rebuild the mask
    python check_binding.py --pipeline --sample_ids 06   CPU, the pipeline's own __call__ through edit(), with a
        tiny random VAE and transformer, the real tokenizer, processor and scheduler config, a causal stub text
        encoder: with no instances, edit() gives the same pixels as the plain pipeline with prompt " " (both
        position modes); with the sample's instances it runs in every mode; the wrappers are removed afterwards
"""
import argparse
import sys
from types import SimpleNamespace

import torch

from rosie.binding import allowed_mask, binding, row_classes
from rosie.tokens import build_layout

N_VIS, SLOT, H, W = 3, 4, 4, 4
OK = True


def report(name, cond, extra=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {extra}")


def synthetic_layout(prompt_pos="shared", which=(0, 1), overlap=False):
    """Vision span of 3, slots of 4 (target words at 2, 3); instance 1 = top-left 2x2 of the 4x4 grid, instance 2 =
    bottom-right 2x2 (overlap: the bottom-right 3x3, sharing one token). which: the instances (and prompts) kept."""
    m0, m1 = torch.zeros(H, W, dtype=torch.bool), torch.zeros(H, W, dtype=torch.bool)
    m0[:2, :2] = True
    m1[1 if overlap else 2:, 1 if overlap else 2:] = True
    locs = [dict(ids=[0] * SLOT, src_words=[], tgt_words=[2, 3]) for _ in range(2)]
    return build_layout(N_VIS, SLOT, [locs[k] for k in which], [[m0, m1][k] for k in which], H, W, prompt_pos)


def classes_rebuild(L, bind=True) -> tuple:
    classes = row_classes(L, bind)
    rows = torch.cat([r for r, _ in classes])
    dense = torch.zeros(L.n, L.n, dtype=torch.bool)
    for r, k in classes:
        dense[r[:, None], k] = True
    once = torch.equal(rows.sort().values, torch.arange(L.n))
    return once and torch.equal(dense, allowed_mask(L, bind)), len(classes)


def tiny_model(num_layers, num_single_layers):
    from longcat.transformer_longcat_image import LongCatImageTransformer2DModel
    torch.manual_seed(0)
    m = LongCatImageTransformer2DModel(in_channels=8, num_layers=num_layers, num_single_layers=num_single_layers,
                                       attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                       pooled_projection_dim=32, axes_dims_rope=[4, 6, 6])
    for p in m.parameters():  # zero-init layers (modulation, gates) would hide differences
        torch.nn.init.normal_(p, std=0.2)
    return m.eval()


def img_ids(L):
    from longcat.pipeline_longcat_image_edit import prepare_pos_ids
    s = (L.img_start, L.img_start)
    return torch.cat([prepare_pos_ids(modality_id=m, type="image", start=s, height=H, width=W) for m in (1, 2)])


def forward(model, L, latents, text, bind=True, t=0.9):
    """The model's output for [target | source] (the pipeline keeps the target part)."""
    x = dict(hidden_states=latents, encoder_hidden_states=text, txt_ids=L.text_ids(), img_ids=img_ids(L),
             timestep=torch.tensor([t]), return_dict=False)
    if bind is None:
        return model(**x)[0][0]
    with binding(model, L, bind):
        return model(**x)[0][0]


def check_tiny():
    HW = H * W
    for overlap in (False, True):
        L = synthetic_layout(overlap=overlap)
        for bind in (True, False):
            exact, n = classes_rebuild(L, bind)
            report(f"row classes cover every row once and rebuild the mask (bind {bind}, overlap {overlap})",
                   exact, f"({n} classes)")
    L = synthetic_layout()
    inst0, inst1 = L.inst
    bg = torch.tensor([g for g in range(HW) if g not in set(inst0.tolist()) | set(inst1.tolist())])
    allowed = allowed_mask(L)
    t0, s0 = L.tgt.start, L.src.start
    for k, sl in enumerate(L.local):
        own = set(range(N_VIS)) | set(range(sl.start, sl.stop)) | set((t0 + L.inst[k]).tolist()) | \
            set((s0 + L.inst[k]).tolist())
        keys = set(allowed[sl].any(0).nonzero().flatten().tolist())
        report(f"slot {k + 1} reads only the vision span, itself and its instance's target and source tokens",
               keys == own)

    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 2 * HW, 8, generator=g)
    text = torch.randn(1, L.n_text, 32, generator=g)
    with torch.no_grad():
        model = tiny_model(2, 2)
        Ls = synthetic_layout("stacked")
        d = (forward(model, Ls, latents, text, None) - forward(model, Ls, latents, text, bind=False)).abs().max()
        report("bind off == the model's processor (2 double + 2 single blocks)", d < 1e-5, f"max|diff|={d:.2g}")

        for name, (nd, ns) in (("double", (1, 0)), ("single", (0, 1))):
            one = tiny_model(nd, ns)
            base = forward(one, L, latents, text)

            def changed(sl):
                t = text.clone()
                t[0, sl] += 3.0
                return (forward(one, L, latents, t) - base).abs().amax(-1)

            d = changed(L.local[1])
            tgt, src = d[:HW], d[HW:]
            report(f"[{name}] slot 2 changed: instance 1 and background unchanged (target and source)",
                   max(tgt[inst0].max(), tgt[bg].max(), src[inst0].max(), src[bg].max()) == 0)
            report(f"[{name}] slot 2 changed: instance 2 changed (target and source)",
                   min(tgt[inst1].min(), src[inst1].min()) > 0)
            d = changed(slice(N_VIS, L.glob.stop))
            tgt, src = d[:HW], d[HW:]
            report(f"[{name}] global slot changed: instance targets unchanged, background and every source changed",
                   tgt[torch.cat([inst0, inst1])].max() == 0 and tgt[bg].min() > 0 and src.min() > 0)
            d = changed(L.vis)
            report(f"[{name}] vision span changed: every target token changed", d[:HW].min() > 0)

            # positions: each instance alone, and no instance prompt at all
            for mode in ("shared", "stacked"):
                Lm = synthetic_layout(mode)
                both = forward(one, Lm, latents, text)[:HW]
                for k in (0, 1):
                    Lk = synthetic_layout(mode, which=(k,))
                    tk = torch.cat([text[:, :Lm.glob.stop], text[:, Lm.local[k]]], 1)
                    d = (both[Lm.inst[k]] - forward(one, Lk, latents, tk)[:HW][Lm.inst[k]]).abs().max()
                    if mode == "shared":
                        report(f"[{name}] shared: instance {k + 1} == a run with only its prompt", d < 1e-5,
                               f"max|diff|={d:.2g}")
                    elif k == 1:
                        report(f"[{name}] stacked: instance 2 != a run with only its prompt (the check can fail)",
                               d > 1e-3, f"max|diff|={d:.2g}")
                L0 = synthetic_layout(mode, which=())
                d = (both[bg] - forward(one, L0, latents, text[:, :Lm.glob.stop])[:HW][bg]).abs().max()
                if mode == "shared":
                    report(f"[{name}] shared: background == a run with no instance prompt", d < 1e-5,
                           f"max|diff|={d:.2g}")

    from longcat.pipeline_longcat_image_edit import prepare_pos_ids
    Ls, Lh = synthetic_layout("stacked"), synthetic_layout("shared")
    report("stacked: txt_ids == the pipeline's, image after all text",
           torch.equal(Ls.text_ids(), prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=Ls.n_text))
           and Ls.img_start == Ls.n_text)
    report("shared: every slot at the global slot's positions, image after the global stream",
           all(torch.equal(Lh.text_pos[sl], Lh.text_pos[N_VIS:Lh.glob.stop]) for sl in Lh.local)
           and Lh.img_start == Lh.glob.stop)


def check_tokens(args):
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    from bench import Bench
    from longcat import MODEL_ID, LongCatImageEditPipeline
    from rosie.instance_edit import instance_prompts, prepare
    from rosie.tokens import GLOBAL_PROMPT, Prompter

    tok = Qwen2Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    proc = Qwen2VLProcessor.from_pretrained(MODEL_ID, subfolder="text_processor")
    pipe = LongCatImageEditPipeline(scheduler=None, vae=None, text_encoder=None, tokenizer=tok,
                                    text_processor=proc, transformer=None)

    def stub(input_ids, attention_mask, pixel_values, image_grid_thw, output_hidden_states):
        """Hidden state at input position j = (j, token id), so the kept tokens can be read back."""
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32)
        return SimpleNamespace(hidden_states=[torch.stack([pos, input_ids[0].float()], -1)[None]])

    pipe.text_encoder = stub  # the pipeline's own _encode_prompt runs on it
    P = Prompter.from_pipe(pipe)
    vs, ve, pad = (tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>", "<|endoftext|>"))
    vis_id = tok.convert_tokens_to_ids("<|image_pad|>")
    bench = Bench(args.dataset_root)
    for sid in args.sample_ids:
        s = bench[sid]
        for mode in ("shared", "stacked"):
            prep = prepare(P, s, prompt_pos=mode)
            L = prep["layout"]
            print(f"{sid} [{mode}] {prep['w']}x{prep['h']} {L.summary()}")
            words = all(tok.decode([ins["ids"][i] for i in ins["tgt_words"]]).strip() == t and
                        tok.decode([ins["ids"][i] for i in ins["src_words"]]).strip() == src
                        for ins, src, t in zip(prep["locs"], s["sources"], s["targets"]))
            report("   source and target words decoded from their positions == the bench's", words)

            image = P.prompt_image(s["image"], prep["w"], prep["h"])
            with instance_prompts(pipe, prep):
                emb, txt_ids = pipe.encode_prompt(prompt=GLOBAL_PROMPT, image=image)
                g = torch.Generator().manual_seed(0)
                _, _, lat_ids, _ = pipe.prepare_latents(None, 1, 16, prep["h"], prep["w"], torch.float32,
                                                        emb.shape[1], torch.device("cpu"), g)
            ids = emb[0, :, 1].long()
            expect = [vs] + [vis_id] * (L.n_vis - 2) + [ve]
            for ins in [dict(ids=tok(GLOBAL_PROMPT, add_special_tokens=False)["input_ids"])] + prep["locs"]:
                expect += ins["ids"] + [pad] * (L.slot_len - len(ins["ids"]))
            report("   every kept token is the layout's (vision span, global slot, instance slots)",
                   ids.tolist() == expect, f"({L.n_text} text tokens, vision_diff {prep['vision_diff']})")
            report("   txt_ids == the layout's, image grid starts at img_start",
                   torch.equal(txt_ids, L.text_ids()) and lat_ids[0].tolist() == [1, L.img_start, L.img_start])
            exact, n = classes_rebuild(L)
            report("   row classes rebuild the mask", exact, f"({n} classes, n = {L.n})")


def tiny_pipeline():
    """The vendored pipeline with a tiny random VAE and transformer (1 double + 1 single block), the real tokenizer,
    VL processor and scheduler config, and a causal stub text encoder (hidden state j = mean of the token embeddings
    up to j)."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    from longcat import MODEL_ID, LongCatImageEditPipeline
    from longcat.transformer_longcat_image import LongCatImageTransformer2DModel

    tok = Qwen2Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    proc = Qwen2VLProcessor.from_pretrained(MODEL_ID, subfolder="text_processor")
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    torch.manual_seed(0)
    vae = AutoencoderKL(down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
                        block_out_channels=[8] * 4, norm_num_groups=4, layers_per_block=1, latent_channels=16,
                        scaling_factor=1.0, shift_factor=0.0).eval()
    D = 32
    transformer = LongCatImageTransformer2DModel(in_channels=64, num_layers=1, num_single_layers=1,
                                                 attention_head_dim=16, num_attention_heads=2,
                                                 joint_attention_dim=D, pooled_projection_dim=D,
                                                 axes_dims_rope=[4, 6, 6]).eval()
    table = torch.randn(len(tok), D, generator=torch.Generator().manual_seed(0))

    def stub(input_ids, attention_mask, pixel_values, image_grid_thw, output_hidden_states):
        e = table[input_ids[0]]
        return SimpleNamespace(hidden_states=[(e.cumsum(0) / torch.arange(1, len(e) + 1)[:, None])[None]])

    pipe = LongCatImageEditPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=tok, text_processor=proc,
                                    transformer=transformer)
    pipe.text_encoder = stub
    pipe.set_progress_bar_config(disable=True)
    return pipe


def check_pipeline(args):
    """The pipeline's own __call__ through rosie.instance_edit.edit, on tiny_pipeline()."""
    import numpy as np

    from bench import Bench
    from rosie.instance_edit import edit
    from rosie.tokens import GLOBAL_PROMPT

    pipe = tiny_pipeline()
    transformer = pipe.transformer
    s = Bench(args.dataset_root)[args.sample_ids[0]]
    steps = 2
    with torch.no_grad():
        plain = pipe(image=s["image"], prompt=GLOBAL_PROMPT, num_inference_steps=steps, guidance_scale=1.0,
                     generator=torch.Generator("cpu").manual_seed(0)).images[0]
        none = dict(s, sources=[], targets=[], masks=[])
        for mode in ("shared", "stacked"):
            out, _ = edit(pipe, none, prompt_pos=mode, steps=steps)
            report(f"no instances [{mode}]: edit() == the plain pipeline with prompt {GLOBAL_PROMPT!r}",
                   np.array_equal(np.asarray(out), np.asarray(plain)), f"({out.size[0]}x{out.size[1]})")
        for mode, bind in (("shared", True), ("stacked", True), ("stacked", False)):
            out, prep = edit(pipe, s, prompt_pos=mode, bind=bind, steps=steps)
            L = prep["layout"]
            report(f"sample {s['sample_id']} [{mode}, bind {bind}] runs: {len(L.local)} instances, n = {L.n}",
                   out.size == (prep["w"], prep["h"]) and prep["vision_diff"] == 0 and
                   not np.array_equal(np.asarray(out), np.asarray(plain)), f"vision_diff {prep['vision_diff']}")
        report("the wrappers are gone afterwards", "encode_prompt" not in vars(pipe) and
               "prepare_latents" not in vars(pipe) and
               all(type(p).__name__ == "LongCatImageAttnProcessor" for p in transformer.attn_processors.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", action="store_true")
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_ids", type=str, nargs="+", default=["06", "09", "276"])
    args = p.parse_args()
    if args.tokens:
        check_tokens(args)
    elif args.pipeline:
        check_pipeline(args)
    else:
        check_tiny()
    print("PASS" if OK else "FAIL")
    sys.exit(0 if OK else 1)
