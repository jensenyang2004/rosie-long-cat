"""
One instance-controlled edit: tokens (rosie/tokens.py) -> layout -> binding (rosie/binding.py), optionally rotation
steering (rosie/steer.py) -> the vendored pipeline, wrapped at runtime (its encode_prompt and prepare_latents, the
attention processors, block hooks), never edited.

edit_single: no binding and no instance prompts. The pipeline's own one-instruction edit (the baseline's call), with
the masks used only by the steering, and instance k's instruction anchor read from its target phrase inside that
instruction.
"""
import inspect
import re
from contextlib import contextmanager, nullcontext

import torch

from bench import output_size
from rosie.binding import binding
from rosie.steer import RotationSteer, steering
from rosie.tokens import (GLOBAL_PROMPT, LOCAL_TEMPLATE, Layout, Prompter, build_layout, build_single_layout,
                          grid_masks)


def prepare(prompter: Prompter, sample: dict, prompt_pos: str = "shared", strict: bool = True) -> dict:
    """Everything that needs no model: generation size, prompts, grid masks, layout."""
    w, h = output_size(*sample["image"].size)
    H, W = h // 16, w // 16
    n_vis = prompter.vision_len(prompter.prompt_image(sample["image"], w, h))
    contents = [LOCAL_TEMPLATE.format(src=s, tgt=t) for s, t in zip(sample["sources"], sample["targets"])]
    locs = [prompter.instruction(c, s, t) for c, s, t in zip(contents, sample["sources"], sample["targets"])]
    layout = build_layout(n_vis, prompter.slot_len, locs, grid_masks(sample["masks"], H, W, strict), H, W,
                          prompt_pos)
    return dict(w=w, h=h, contents=contents, locs=locs, layout=layout)


@contextmanager
def instance_prompts(pipe, prep: dict):
    """Replace the pipeline's encode_prompt: the global prompt and every instance prompt, each encoded on its own by
    pipe._encode_prompt with the pipeline's own prompt image; the vision span kept once, the slots appended; text
    positions from the layout. With shared positions, also move the image grid's start (prepare_latents) to where it
    is in a one-prompt edit. prep["vision_diff"] gets the largest |difference| between the global prompt's vision
    span and an instance prompt's (0 expected: the encoder is causal)."""
    L: Layout = prep["layout"]
    prep["vision_diff"] = 0.0
    latents = pipe.prepare_latents
    latents_sig = inspect.signature(latents)

    def encode_prompt(prompt=None, image=None, num_images_per_prompt=1, prompt_embeds=None):
        assert num_images_per_prompt == 1 and prompt_embeds is None and image is not None
        assert prompt in (GLOBAL_PROMPT, [GLOBAL_PROMPT]), "classifier-free guidance is not supported"
        g = pipe._encode_prompt([GLOBAL_PROMPT], image)
        assert g.shape[1] == L.glob.stop, f"global stream of {g.shape[1]} tokens, layout {L.glob.stop}"
        parts = [g]
        for content in prep["contents"]:
            e = pipe._encode_prompt([content], image)
            assert e.shape[1] == L.glob.stop, f"instance prompt {content!r}: {e.shape[1]} tokens"
            prep["vision_diff"] = max(prep["vision_diff"], (e[:, L.vis] - g[:, L.vis]).abs().max().item())
            parts.append(e[:, L.n_vis:])
        emb = torch.cat(parts, dim=1)
        assert emb.shape[1] == L.n_text
        return emb, L.text_ids().to(emb.device)

    def prepare_latents(*args, **kwargs):
        a = latents_sig.bind(*args, **kwargs)
        assert a.arguments["prompt_embeds_length"] == L.n_text
        a.arguments["prompt_embeds_length"] = L.img_start
        return latents(*a.args, **a.kwargs)

    pipe.encode_prompt = encode_prompt  # instance attributes shadow the methods
    pipe.prepare_latents = prepare_latents
    try:
        yield
    finally:
        del pipe.encode_prompt, pipe.prepare_latents


def edit(pipe, sample: dict, prompt_pos: str = "shared", strict: bool = True, bind: bool = True, seed: int = 0,
         steps: int = 8, steer: RotationSteer | None = None):
    """Returns (PIL image at the generation size, prep). Guidance 1 only (Turbo): with CFG the pipeline would also
    encode a negative prompt, which this does not lay out. steer: rotation steering; its per-(step, block) stats are
    in steer.stats afterwards."""
    if prompt_pos == "shared" and not bind:
        raise ValueError("shared prompt positions need the binding (otherwise rows read slots at equal positions)")
    prep = prepare(Prompter.from_pipe(pipe), sample, prompt_pos, strict)
    steered = nullcontext() if steer is None else steering(pipe.transformer, steer, prep["layout"], pipe.scheduler)
    with instance_prompts(pipe, prep), binding(pipe.transformer, prep["layout"], bind), steered:
        out = pipe(image=sample["image"], prompt=GLOBAL_PROMPT, num_inference_steps=steps, guidance_scale=1.0,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    return out, prep


def target_phrase(target: str) -> str:
    """The bench's fg_prompt as it is looked up in edit_inst_single: no leading article, no end punctuation."""
    return re.sub(r"^(a|an|the)\s+", "", target.strip().rstrip(".,;:!"), flags=re.IGNORECASE)


def prepare_single(prompter: Prompter, sample: dict, strict: bool = True, prompt_key: str = "edit_inst_single"):
    w, h = output_size(*sample["image"].size)
    H, W = h // 16, w // 16
    n_vis = prompter.vision_len(prompter.prompt_image(sample["image"], w, h))
    content = sample[prompt_key]
    phrases = [target_phrase(t) for t in sample["targets"]]
    ph = prompter.phrases(content, phrases)
    layout = build_single_layout(n_vis, prompter.slot_len, ph["words"], grid_masks(sample["masks"], H, W, strict),
                                 H, W)
    return dict(w=w, h=h, content=content, ids=ph["ids"], phrases=phrases,
                found=[len(x) > 0 for x in ph["words"]], layout=layout)


def edit_single(pipe, sample: dict, strict: bool = True, seed: int = 0, steps: int = 8,
                steer: RotationSteer | None = None, prompt_key: str = "edit_inst_single"):
    """The baseline's call (infer_baseline.py: one instruction, guidance 1), with steering hooks only. An instance
    whose target phrase is not in the instruction gets no instruction term (empty anchor); the other terms need only
    its mask. Returns (PIL image at the generation size, prep)."""
    prep = prepare_single(Prompter.from_pipe(pipe), sample, strict, prompt_key)
    steered = nullcontext() if steer is None else steering(pipe.transformer, steer, prep["layout"], pipe.scheduler)
    with steered:
        out = pipe(image=sample["image"], prompt=prep["content"], negative_prompt="", num_inference_steps=steps,
                   guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    return out, prep
