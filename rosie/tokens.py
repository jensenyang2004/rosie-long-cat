"""
Token assignment for the joint LongCat-Image-Edit sequence. Pure bookkeeping: needs the tokenizer and the VL image
processor, never the model.

The pipeline encodes a prompt with the source image (_encode_prompt) into a text stream of fixed shape:

    [vision span: <|vision_start|>, N_v image tokens, <|vision_end|>] [instruction slot: the instruction, then
    padding, 512 tokens]

The encoder (Qwen2.5-VL) is causal and the vision span comes before the instruction, so the vision span is the same
for every prompt of one image. We encode the global prompt and each instance prompt ("Replace the {src} with
{tgt}", mice's outer_local_prompts) this way, keep the vision span once and every instruction slot whole (padding
included, as the transformer always reads it), and the joint sequence becomes

    [vision span | global slot] [slot 1] ... [slot K] [target image, H*W] [source image, H*W]
    (the global stream is the vision span and the global slot)

with target before source, as the pipeline concatenates them. Source token i and target token i share a grid
position (the source is resized to the generation size).

Text RoPE positions (the pipeline's prepare_pos_ids): text token at index p is at (0, p, p), on the diagonal of the
image axes; the image grid starts at (T, T) with T the text length.

    stacked  what the pipeline does with a longer text: slot k follows slot k-1, and the image grid starts after
             all of them (512 positions later per instance)
    shared   every slot at the global slot's positions, and the image grid where it starts in a one-prompt edit.
             Each token then sees its prompt and the image at the positions of an ordinary edit. Needs the binding:
             with it the only rows that read two slots are the source tokens of an instance (global + own slot)
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LOCAL_TEMPLATE = "Replace the {src} with {tgt}"
GLOBAL_PROMPT = " "  # mice's global prompt
POSITIONS = ("shared", "stacked")


@dataclass
class Prompter:
    """What the pipeline uses to encode a prompt: tokenizer, VL image processor, slot length, resizing."""
    tokenizer: object
    image_processor_vl: object
    image_processor: object  # the VAE-side resize (pipe.image_processor)
    slot_len: int  # pipe.tokenizer_max_length

    @classmethod
    def from_pipe(cls, pipe):
        return cls(pipe.tokenizer, pipe.image_processor_vl, pipe.image_processor, pipe.tokenizer_max_length)

    def prompt_image(self, image: Image.Image, w: int, h: int) -> Image.Image:
        """The image the text encoder sees, made as in the pipeline's __call__ (generation size, then half)."""
        image = self.image_processor.resize(image, h, w)
        return self.image_processor.resize(image, h // 2, w // 2)

    def vision_len(self, prompt_image: Image.Image) -> int:
        """Length of the vision span: the image tokens (as _encode_prompt counts them) and the two markers."""
        grid = self.image_processor_vl(images=prompt_image, return_tensors="pt")["image_grid_thw"]
        return int(grid.prod()) // self.image_processor_vl.merge_size ** 2 + 2

    def instruction(self, content: str, src: str | None = None, tgt: str | None = None) -> dict:
        """The slot's tokens as _encode_prompt makes them (one chunk: no quotes, so split_quotation keeps the
        prompt whole), with the positions of the source and target words in the slot (target = after
        "<src> with ", as fixed in mice)."""
        from longcat.pipeline_longcat_image_edit import split_quotation
        assert split_quotation(content) == [(content, False)], f"quoted text is tokenized per character: {content!r}"
        tok = self.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(tok["input_ids"]), list(tok["offset_mapping"])
        assert len(ids) <= self.slot_len, f"instruction longer than the slot: {content!r}"

        def span(start, end):
            return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]

        src_words, tgt_words = [], []
        if tgt is not None:
            prefix, sep = "Replace the ", " with "
            if src is not None and content.startswith(prefix + src + sep):
                s0 = len(prefix)
                t0 = s0 + len(src) + len(sep)
            else:
                s0, t0 = content.find(src or ""), content.rfind(tgt)
            if src:
                src_words = span(s0, s0 + len(src))
            tgt_words = span(t0, t0 + len(tgt))
        return dict(ids=ids, src_words=src_words, tgt_words=tgt_words)

    def phrases(self, content: str, phrases: list) -> dict:
        """The slot's tokens for one free-form instruction (the bench's edit_inst_single), and per phrase the slot
        positions of every whole-word, case-insensitive occurrence of it, plural -s/-es allowed ([] if it does not
        occur in the slot). Paraphrases are not matched: a shorter fallback (the phrase's last words) maps different
        targets to the same words ("fork" for two forks)."""
        import re
        ids = self.instruction(content)["ids"]
        offsets = self.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        words = []
        for ph in phrases:
            pat = r"\b" + re.escape(ph) + r"(?:s|es)?\b"
            spans = [m.span() for m in re.finditer(pat, content, re.IGNORECASE)] if ph else []
            words.append([i for i, (a, b) in enumerate(offsets) if b > a and any(a < e and b > s for s, e in spans)])
        return dict(ids=ids, words=words)


def grid_masks(masks: list, H: int, W: int, strict: bool = True) -> list:
    """Instance masks on the H x W latent grid (nearest, as mice's resize_mask). strict: each mask loses the tokens
    also claimed by any smaller instance (mice's --strict_masks)."""
    g = [F.interpolate(torch.as_tensor(np.asarray(m), dtype=torch.float32)[None, None], size=(H, W),
                       mode="nearest")[0, 0] > 0.5 for m in masks]
    if strict and len(g) > 1:
        areas = [int(m.sum()) for m in g]
        carved = []
        for i, m in enumerate(g):
            m = m.clone()
            for j, o in enumerate(g):
                if j != i and areas[j] < areas[i]:
                    m &= ~o
            carved.append(m)
        g = carved
    return g


@dataclass
class Layout:
    """Positions in the joint sequence (what the attention processor and the steering index)."""
    n: int  # joint length
    H: int
    W: int
    n_vis: int  # vision span length
    slot_len: int
    vis: slice  # vision span (part of the global stream; every prompt's stream starts with it)
    glob: slice  # global stream: vision span + global slot
    local: list  # per instance: slice of its slot
    tgt: slice  # target image block (H*W, row-major)
    src: slice  # source image block
    inst: list  # per instance: flat grid indices (strict-carved), the same for source and target
    src_words: list  # per instance: joint positions of the source words
    tgt_words: list  # per instance: joint positions of the target words
    prompt_pos: str
    text_pos: torch.Tensor  # (n_text,) RoPE index p of each text token, at (0, p, p)
    img_start: int  # the image grid starts at (img_start, img_start)

    @property
    def n_text(self) -> int:
        return self.tgt.start

    def text_ids(self) -> torch.Tensor:
        """(n_text, 3) txt_ids as the pipeline's prepare_pos_ids makes them (float, modality 0)."""
        p = self.text_pos.float()
        return torch.stack([torch.zeros_like(p), p, p], dim=1)

    def summary(self) -> dict:
        return dict(n=self.n, grid=[self.H, self.W], n_vis=self.n_vis, slot_len=self.slot_len,
                    n_text=self.n_text, instances=len(self.inst), inst=[len(i) for i in self.inst],
                    tgt_words=[len(w) for w in self.tgt_words], prompt_pos=self.prompt_pos, img_start=self.img_start)


def build_layout(n_vis: int, slot_len: int, locals_: list, masks: list, H: int, W: int,
                 prompt_pos: str = "shared") -> Layout:
    """n_vis: Prompter.vision_len; locals_: Prompter.instruction() of each instance prompt; masks: grid_masks()."""
    assert prompt_pos in POSITIONS, prompt_pos
    assert len(locals_) == len(masks), f"{len(locals_)} instance prompts, {len(masks)} masks (Bench.paired)"
    n_glob = n_vis + slot_len
    local, src_words, tgt_words = [], [], []
    for k, ins in enumerate(locals_):
        start = n_glob + k * slot_len
        local.append(slice(start, start + slot_len))
        src_words.append(torch.tensor([start + i for i in ins["src_words"]], dtype=torch.long))
        tgt_words.append(torch.tensor([start + i for i in ins["tgt_words"]], dtype=torch.long))
    n_text = n_glob + len(locals_) * slot_len
    HW = H * W
    tgt, src = slice(n_text, n_text + HW), slice(n_text + HW, n_text + 2 * HW)
    if prompt_pos == "stacked":
        text_pos, img_start = torch.arange(n_text), n_text
    else:
        text_pos = torch.cat([torch.arange(n_glob)] + [torch.arange(n_vis, n_glob)] * len(locals_))
        img_start = n_glob
    inst = [m.reshape(-1).nonzero(as_tuple=True)[0] for m in masks]
    return Layout(src.stop, H, W, n_vis, slot_len, slice(0, n_vis), slice(0, n_glob), local, tgt, src, inst,
                  src_words, tgt_words, prompt_pos, text_pos, img_start)


def build_single_layout(n_vis: int, slot_len: int, words: list, masks: list, H: int, W: int) -> Layout:
    """The pipeline's own sequence for one instruction, [vision span | instruction slot] [target] [source] at its own
    positions, for steering without binding (no instance slots). words: Prompter.phrases()["words"], instance k's
    target words in the slot; masks: grid_masks()."""
    assert len(words) == len(masks), f"{len(words)} target phrases, {len(masks)} masks (Bench.paired)"
    n_glob = n_vis + slot_len
    HW = H * W
    tgt, src = slice(n_glob, n_glob + HW), slice(n_glob + HW, n_glob + 2 * HW)
    tgt_words = [torch.tensor([n_vis + i for i in w], dtype=torch.long) for w in words]
    inst = [m.reshape(-1).nonzero(as_tuple=True)[0] for m in masks]
    return Layout(src.stop, H, W, n_vis, slot_len, slice(0, n_vis), slice(0, n_glob), [], tgt, src, inst,
                  [torch.zeros(0, dtype=torch.long)] * len(masks), tgt_words, "single", torch.arange(n_glob), n_glob)
