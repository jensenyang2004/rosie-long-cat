"""
Check that the vendored LongCat-Image-Edit code computes exactly what the installed diffusers does.

    python check_vendored.py              CPU, no weights: (1) each vendored file equals the installed one except its
                                          import lines; (2) a tiny random transformer, the installed class vs the
                                          vendored one with the same weights, on [text | target | source]. Equal.
    python check_vendored.py --processor  CPU, tokenizer and processor files only (no weights): the pipeline's own
                                          prompt encoding on one bench image, with a stub text encoder. Prints what
                                          the text stream holds (vision tokens, instruction, padding).
    python check_vendored.py --full       GPU: the installed LongCatImageEditPipeline vs load_pipeline() on one bench
                                          image, same seed; saves both and prints the max pixel difference (0).
"""
import argparse
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HEADER_LINES = 5  # the "Vendored from ..." comment block on top of each vendored file


def check_source() -> bool:
    import diffusers.models.transformers.transformer_longcat_image as ref_t
    import diffusers.pipelines.longcat_image.pipeline_longcat_image_edit as ref_p
    import longcat.pipeline_longcat_image_edit as ven_p
    import longcat.transformer_longcat_image as ven_t

    ok = True
    for ref, ven in ((ref_t, ven_t), (ref_p, ven_p)):
        a = Path(inspect.getsourcefile(ref)).read_text().splitlines()
        b = Path(inspect.getsourcefile(ven)).read_text().splitlines()[HEADER_LINES:]
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        same = len(a) == len(b) and all(x.startswith("from ") and y.startswith("from ") for x, y in diff)
        ok &= same
        print(f"{Path(ven.__file__).name}: {len(a)} lines, {len(diff)} import lines changed, rest equal={same}")
    return ok


def check_tiny() -> bool:
    from diffusers.models.transformers.transformer_longcat_image import LongCatImageTransformer2DModel as Ref
    from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids

    from longcat.transformer_longcat_image import LongCatImageTransformer2DModel as Ven

    assert Ref is not Ven
    cfg = dict(in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=16, num_attention_heads=2,
               joint_attention_dim=32, pooled_projection_dim=32, axes_dims_rope=[4, 6, 6])
    torch.manual_seed(0)
    ref = Ref(**cfg).eval()
    ven = Ven(**cfg).eval()
    for p in ref.parameters():  # zero-init layers (modulation, gates) would hide differences
        torch.nn.init.normal_(p, std=0.2)
    ven.load_state_dict(ref.state_dict())

    # the pipeline's layout: text (modality 0), target (1) and source (2) on the same grid, after the text
    g = torch.Generator().manual_seed(1)
    L, H, W = 5, 4, 4
    txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=L)
    img_ids = torch.cat([prepare_pos_ids(modality_id=m, type="image", start=(L, L), height=H, width=W)
                         for m in (1, 2)])
    x = dict(hidden_states=torch.randn(1, 2 * H * W, 8, generator=g),
             encoder_hidden_states=torch.randn(1, L, 32, generator=g), txt_ids=txt_ids, img_ids=img_ids,
             return_dict=False)
    ok = True
    with torch.no_grad():
        for t in (0.9, 0.3):
            a = ref(**x, timestep=torch.tensor([t]))[0]
            b = ven(**x, timestep=torch.tensor([t]))[0]
            same = torch.equal(a, b)
            ok &= same
            print(f"tiny transformer t={t}: shape {tuple(a.shape)}, equal={same}, "
                  f"max|diff|={(a - b).abs().max().item():.3g}, |out|={a.abs().mean().item():.3g}")
    return ok


def check_processor(args) -> bool:
    """Runs the vendored pipeline's _encode_prompt with the real tokenizer and VL processor and a stub text encoder
    whose hidden state at input position j is j, so the kept positions can be read back."""
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    from bench import Bench, output_size
    from longcat import MODEL_ID, LongCatImageEditPipeline

    tok = Qwen2Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    proc = Qwen2VLProcessor.from_pretrained(MODEL_ID, subfolder="text_processor")
    pipe = LongCatImageEditPipeline(scheduler=None, vae=None, text_encoder=None, tokenizer=tok,
                                    text_processor=proc, transformer=None)
    seen = {}

    def stub(input_ids, attention_mask, pixel_values, image_grid_thw, output_hidden_states):
        seen.update(ids=input_ids[0].cpu(), mask=attention_mask[0].cpu(), grid=image_grid_thw.cpu())
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32)
        return SimpleNamespace(hidden_states=[pos[None, :, None].expand(1, -1, 4)])

    s = Bench(args.dataset_root)[args.sample_id]
    w, h = output_size(*s["image"].size)
    image = pipe.image_processor.resize(s["image"], h, w)
    prompt_image = pipe.image_processor.resize(image, h // 2, w // 2)  # as in __call__
    ns = SimpleNamespace(**{k: getattr(pipe, k) for k in ("image_processor_vl", "tokenizer", "tokenizer_max_length",
                                                           "prompt_template_encode_prefix",
                                                           "prompt_template_encode_suffix", "image_token")},
                         text_encoder=stub, device=torch.device("cpu"))
    emb = LongCatImageEditPipeline._encode_prompt(ns, [s["edit_inst_single"]], prompt_image)
    kept = emb[0, :, 0].long()
    ids, mask = seen["ids"][kept], seen["mask"][kept]
    vis_id = tok.convert_tokens_to_ids("<|image_pad|>")
    n_vis = int((ids == vis_id).sum())
    real = mask.bool() & (ids != vis_id)
    pad = ~mask.bool()
    first_pad = int(pad.nonzero()[0]) if pad.any() else None
    print(f"sample {args.sample_id}: source {s['image'].size} -> generation {w}x{h} (grid {h // 16}x{w // 16}), "
          f"VLM image {prompt_image.size}, grid_thw {seen['grid'].tolist()}")
    print(f"text stream: {len(kept)} tokens = {n_vis} vision + {int(real.sum())} other unmasked + {int(pad.sum())} "
          f"padding (tokenizer padding_side={tok.padding_side}, first padding at {first_pad})")
    print("starts:", repr(tok.decode(ids[:3])), "...", "instruction:", repr(tok.decode(ids[real][2:14])), "...")
    print("ends:", repr(tok.decode(ids[-3:])))
    contiguous = kept.tolist() == list(range(int(kept[0]), int(kept[0]) + len(kept)))
    print(f"kept input positions contiguous: {contiguous}")
    return contiguous and n_vis > 0


def check_full(args) -> bool:
    import numpy as np
    from diffusers import LongCatImageEditPipeline as Official

    from bench import Bench
    from longcat import MODEL_ID, SAMPLING, load_pipeline

    s = Bench(args.dataset_root)[args.sample_id]
    steps, guidance = SAMPLING[MODEL_ID]
    call = dict(image=s["image"], prompt=s["edit_inst_single"], negative_prompt="", num_inference_steps=steps,
                guidance_scale=guidance)
    out = {}
    for name in ("official", "vendored"):
        if name == "official":
            pipe = Official.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload(device=args.device) if args.cpu_offload else pipe.to(args.device)
        else:
            pipe = load_pipeline(MODEL_ID, args.device, cpu_offload=args.cpu_offload)
        print(name, type(pipe.transformer).__module__)
        out[name] = pipe(**call, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        out[name].save(f"check_{name}_{args.sample_id}.png")
        del pipe
        torch.cuda.empty_cache()
    d = np.abs(np.asarray(out["official"], np.int16) - np.asarray(out["vendored"], np.int16)).max()
    print(f"max pixel difference: {d}  (check_official_{args.sample_id}.png, check_vendored_{args.sample_id}.png)")
    return d == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--processor", action="store_true")
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_id", type=str, default="06")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    args = p.parse_args()
    if args.full:
        ok = check_full(args)
    elif args.processor:
        ok = check_processor(args)
    else:
        ok = check_source()
        ok = check_tiny() and ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
