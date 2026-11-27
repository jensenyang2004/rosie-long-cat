"""
Text binding for LongCat-Image-Edit: mice's final binding (apitasmkernelnonlap + free_latent + free_context, strict
masks, every block, every step) on the joint sequence of rosie/tokens.py.

A prompt here is its whole stream: the vision span and its slot. Allowed (query -> key):

    global stream (vision span, global slot) -> global stream, every image token
    instance slot k      -> vision span, slot k, the target and source tokens of instance k
    target, background   -> global stream, every image token
    target, instance k   -> vision span, slot k, every image token
    source, background   -> global stream, every image token
    source, instance k   -> global stream, slot k, every image token

Image-to-image attention is never restricted (free_latent + free_context). As in mice, an instance's target tokens
never read the global slot, its source tokens read the global slot and their own, and a grid token claimed by two
instances (only with --no_strict, or equal areas) reads both slots.

The vision span has one copy, in the global stream. As a key it belongs to every prompt (it is the same in every
prompt's encoding); as a query it is the global prompt's.

The (n, n) mask is never built for a run: the rows above fall into a few classes, each reading one key set, so
attention runs per class on just its keys, unmasked (any backend, flash included). allowed_mask() is the dense form,
written from the rules independently, for the checks.
"""
from contextlib import contextmanager

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

from longcat.transformer_longcat_image import _get_qkv_projections
from rosie.tokens import Layout


def _membership(L: Layout) -> torch.Tensor:
    """(K, H*W) bool: grid token -> instances that claim it."""
    m = torch.zeros(len(L.inst), L.H * L.W, dtype=torch.bool)
    for k, idx in enumerate(L.inst):
        m[k, idx] = True
    return m


def row_classes(L: Layout, bind: bool = True) -> list:
    """[(query rows, key positions)]: every joint position is a row of exactly one class. bind=False: one class,
    everything reads everything (the model's own attention)."""
    n = L.n
    every = torch.arange(n)
    if not bind:
        return [(every, every)]

    def keys(*parts):
        k = torch.zeros(n, dtype=torch.bool)
        for p in parts:
            k[p] = True
        return k.nonzero().flatten()

    img, glob, vis = slice(L.tgt.start, n), L.glob, L.vis
    t0, s0 = L.tgt.start, L.src.start
    classes = []
    for k, (sl, idx) in enumerate(zip(L.local, L.inst)):
        classes.append((torch.arange(sl.start, sl.stop), keys(vis, sl, t0 + idx, s0 + idx)))
    member = _membership(L)
    groups = {}  # grid tokens by the set of instances that claim them
    for g in range(L.H * L.W):
        groups.setdefault(tuple(member[:, g].nonzero().flatten().tolist()), []).append(g)
    background = torch.tensor(groups.pop((), []), dtype=torch.long)
    classes.insert(0, (torch.cat([torch.arange(glob.start, glob.stop), t0 + background, s0 + background]),
                       keys(glob, img)))
    for ks, grid in groups.items():
        grid = torch.tensor(grid, dtype=torch.long)
        slots = [L.local[k] for k in ks]
        classes.append((t0 + grid, keys(vis, *slots, img)))
        classes.append((s0 + grid, keys(glob, *slots, img)))
    return [(r, k) for r, k in classes if len(r)]


def allowed_mask(L: Layout, bind: bool = True) -> torch.Tensor:
    """(n, n) bool, True = may attend: the rules of the module docstring, for the checks."""
    n = L.n
    if not bind:
        return torch.ones(n, n, dtype=torch.bool)
    pos = torch.arange(n)
    is_vis = pos < L.vis.stop
    is_glob = pos < L.glob.stop
    is_slot = (pos >= L.glob.stop) & (pos < L.n_text)
    is_tgt = (pos >= L.tgt.start) & (pos < L.tgt.stop)
    is_src = pos >= L.src.start
    is_img = is_tgt | is_src
    member = torch.zeros(len(L.local), n, dtype=torch.bool)  # slot k, and the target/source tokens of instance k
    for k, (sl, idx) in enumerate(zip(L.local, L.inst)):
        member[k, sl] = True
        member[k, L.tgt.start + idx] = True
        member[k, L.src.start + idx] = True
    share = torch.zeros(n, n, dtype=torch.bool)  # query and key belong to a common instance
    for m in member:
        share |= m[:, None] & m[None, :]
    in_inst = member.any(0)
    own_slot = share & is_slot[None, :]
    rule = is_glob[:, None] & (is_glob | is_img)[None, :]
    rule |= is_slot[:, None] & (is_vis[None, :] | (share & (is_slot | is_img)[None, :]))
    rule |= (is_img & ~in_inst)[:, None] & (is_glob | is_img)[None, :]
    rule |= (is_tgt & in_inst)[:, None] & (is_vis[None, :] | own_slot | is_img[None, :])
    rule |= (is_src & in_inst)[:, None] & (is_glob[None, :] | own_slot | is_img[None, :])
    return rule


def _take(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x[:, idx], a view when idx is a contiguous range."""
    if len(idx) and int(idx[-1]) - int(idx[0]) + 1 == len(idx):
        return x[:, int(idx[0]):int(idx[-1]) + 1]
    return x[:, idx]


class BindAttnProcessor:
    """LongCatImageAttnProcessor (same projections, norms, RoPE and outputs) with attention per row class
    (row_classes). Double blocks get (image, text) and return both; single blocks get [text | image] and return the
    attention output alone, as the model's processor does."""
    _attention_backend = None
    _parallel_config = None

    def __init__(self, classes: list, n: int):
        self.classes, self.n = classes, n
        self._dev = {}

    def _on(self, device):
        if device not in self._dev:
            self._dev[device] = [(r.to(device), k.to(device)) for r, k in self.classes]
        return self._dev[device]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        assert attention_mask is None, "the transformer passes no attention mask"
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if attn.added_kv_proj_dim is not None:
            encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
            encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if query.shape[1] != self.n:
            raise ValueError(f"sequence of {query.shape[1]} tokens, layout of {self.n}")

        out = torch.empty_like(query)
        for rows, keys in self._on(query.device):
            out[:, rows] = dispatch_attention_fn(
                _take(query, rows), _take(key, keys), _take(value, keys), attn_mask=None,
                backend=self._attention_backend, parallel_config=self._parallel_config)
        hidden_states = out.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


@contextmanager
def binding(transformer, L: Layout, bind: bool = True):
    """Install BindAttnProcessor on every block for one sample, then put the previous processors back."""
    previous = transformer.attn_processors
    transformer.set_attn_processor(BindAttnProcessor(row_classes(L, bind), L.n))
    try:
        yield
    finally:
        transformer.set_attn_processor(previous)
