"""
One instance-controlled edit: tokens (rosie/tokens.py) -> layout -> binding (rosie/binding.py) -> the vendored
pipeline, wrapped at runtime (its encode_prompt and prepare_latents, the attention processors), never edited.
"""
import inspect
from contextlib import contextmanager

import torch

from bench import output_size
from rosie.binding import binding
from rosie.tokens import GLOBAL_PROMPT, LOCAL_TEMPLATE, Layout, Prompter, build_layout, grid_masks


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
         steps: int = 8):
    """Returns (PIL image at the generation size, prep). Guidance 1 only (Turbo): with CFG the pipeline would also
    encode a negative prompt, which this does not lay out."""
    if prompt_pos == "shared" and not bind:
        raise ValueError("shared prompt positions need the binding (otherwise rows read slots at equal positions)")
    prep = prepare(Prompter.from_pipe(pipe), sample, prompt_pos, strict)
    with instance_prompts(pipe, prep), binding(pipe.transformer, prep["layout"], bind):
        out = pipe(image=sample["image"], prompt=GLOBAL_PROMPT, num_inference_steps=steps, guidance_scale=1.0,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    return out, prep
