"""
RoSIE's rotation steering (mice flux2/residual_rotate.py, the final config of 4B_ROT_mice_2 = the paper's Ours) on
LongCat-Image-Edit.

In every block, the update of the target image's tokens, D_i = h_out_i - h_in_i, is split into the frame mean and
d_i = D_i - mean_i(D). Only d_i is changed, and only its direction: it is turned on its sphere, length kept, by

  contact      away from each neighbour j's write g_j     instance tokens   theta = -rot_ctc w_kj(i) asin([c_j]_+)
  source       away from the source write s_i             instance tokens   theta = -rot_src asin([c_s]_+)
  background   toward s_i                                 background        theta = +rot_bg omega(i) psi_s
  instruction  toward instance k's target-word write u_k  instance k        theta = +rot_ins psi_u

c = cos(d_i, anchor), psi = angle(d_i, anchor). The tangents are summed, capped at max_turn and applied with the
sphere's exponential map. The terms in sigma_terms are scaled by the step's sigma (mice: f_i = sigma_i).

Anchors, all from the same block's own updates in the same forward (no cache on LongCat, so as in mice):
  g_j  normalize(mean over instance j of d): the target's updates.
  s_i  normalize(centred update of the source token at the same grid position i): the image stream's source part.
  u_k  normalize(mean update of instance k's target-word tokens), raw (mice --sem_ref target): the text stream.
       ins_centre "prompts": minus the mean update over every instruction slot (global and instances, padding
       included); "text": minus the mean over the whole text stream (vision span too). Raw by default.

Both block types take (hidden_states = [target | source], encoder_hidden_states = text) and return (text, image),
so one hook serves the 10 double and the 20 single blocks. mice steered every block of klein (5 double, 20 single)
and read u_k from the text stream of each (its double-block strength rot_ins_double = rot_ins = 0.2).

The math below (rotate and its helpers) is rosie-qwen's, which check_steer.py compares with mice's
ResidualRotate._edit; the hooks are attached at runtime (steering()), so the model code is not edited.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rosie.tokens import Layout

TERMS = ("ctc", "src", "bg", "ins")


def _toward(dh: torch.Tensor, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """theta * (unit tangent at dh toward r). dh, r: [..., D] unit; theta: [..., 1] (negative = away)."""
    c = (dh * r).sum(-1, keepdim=True)
    return theta * (r - c * dh) / (1 - c * c).clamp_min(1e-12).sqrt()


def _deg(v: torch.Tensor, sel: torch.Tensor) -> float:
    """Mean tangent length (deg) of v [B, N, D] over the tokens of sel [N] where it is non-zero."""
    n = v[0, sel].norm(dim=-1)
    n = n[n > 0]
    return float(n.mean()) * 180 / math.pi if n.numel() else 0.0


def pair_weights(masks: torch.Tensor, H: int, W: int, local_tau: float) -> list:
    """Per instance k: [N_k, K] weight of each of k's tokens toward every instance j, exp(-max(dist_j - 1, 0) /
    local_tau) from the token's own distance to j's nearest token; 0 toward k itself."""
    K = masks.shape[0]
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
    dist = torch.full((K, H * W), float("inf"))
    for j in range(K):
        if masks[j].any():
            dist[j] = torch.cdist(coords, coords[masks[j]]).min(dim=1).values
    weights = []
    for k in range(K):
        w = torch.exp(-(dist[:, masks[k]].T - 1.0).clamp_min(0.0) / local_tau)
        w[:, k] = 0.0
        weights.append(w)
    return weights


def bg_weights(masks: torch.Tensor, H: int, W: int, margin: float, tau: float) -> torch.Tensor:
    """[HW] omega(i): 0 on instance tokens and within `margin` of them, then 1 - exp(-(dist - margin) / tau)."""
    inst = masks.any(dim=0)
    if not bool(inst.any()):
        return (~inst).float()
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
    over = (torch.cdist(coords, coords[inst]).min(dim=1).values - margin).clamp_min(0.0)
    w = (over > 0).float() if tau <= 0 else 1.0 - torch.exp(-over / tau)
    return w * (~inst).float()


def source_direction(D_src: torch.Tensor) -> torch.Tensor:
    """s_i: the source tokens' updates [B, HW, D], centred over the image and normalized per token."""
    s = D_src.float()
    return F.normalize(s - s.mean(dim=1, keepdim=True), dim=-1)


def instruction_directions(D_text: torch.Tensor, words: list) -> torch.Tensor:
    """u_k: [B, K, D] normalize(mean update of instance k's target-word tokens); words[k] = positions in D_text."""
    U = torch.zeros(D_text.shape[0], len(words), D_text.shape[-1], device=D_text.device)
    for k, pos in enumerate(words):
        if len(pos):
            U[:, k] = F.normalize(D_text[:, pos].float().mean(dim=1), dim=-1)
    return U


def rotate(h_in: torch.Tensor, h_out: torch.Tensor, masks: torch.Tensor, pair_w: list, bg_w: torch.Tensor,
           s_dir: torch.Tensor | None, U: torch.Tensor | None, ctc: float, src: float, bg: float, ins: float,
           max_turn: float) -> tuple:
    """One block. h_in, h_out: [B, HW, D] target tokens; masks [K, HW] bool; s_dir [B, HW, D] unit; U [B, K, D].
    ctc, src, bg, ins: this step's strengths (0 = off). Returns (edited h_out, stats)."""
    K = masks.shape[0]
    inst = masks.any(dim=0)
    mask_w = masks.float()
    hi, ho = h_in.float(), h_out.float()
    D = ho - hi
    mean = D.mean(dim=1, keepdim=True)
    d = D - mean
    norm = d.norm(dim=-1, keepdim=True)
    dh = d / norm.clamp_min(1e-12)
    v = torch.zeros_like(d)
    stat = {}

    if ctc > 0:  # away from what each neighbour writes, weighted by the token's proximity to it
        cnt = mask_w.sum(dim=1).clamp_min(1.0)[None, :, None]
        g = F.normalize(torch.einsum("kn,bnd->bkd", mask_w, d) / cnt, dim=-1)  # [B, K, D]
        vc = torch.zeros_like(d)
        cover = torch.zeros(masks.shape[1], device=d.device)
        for k in range(K):
            w = pair_w[k]
            if float(w.sum()) == 0.0:
                continue
            idx = masks[k]
            dk = dh[:, idx]
            c = torch.einsum("bnd,bjd->bnj", dk, g)
            coef = -ctc * w[None] * torch.asin(c.clamp(0.0, 1.0)) / (1 - c * c).clamp_min(1e-12).sqrt()
            vc[:, idx] += torch.einsum("bnj,bjd->bnd", coef, g) - (coef * c).sum(-1, keepdim=True) * dk
            cover[idx] += 1.0
        vc = vc / cover.clamp_min(1.0)[None, :, None]
        v += vc
        stat.update(rot_ctc=ctc, ctc_deg=_deg(vc, inst))

    if s_dir is not None:  # instance tokens away from the source write, background toward it
        s_dir = s_dir.float()
        cs = (dh * s_dir).sum(-1, keepdim=True)
        mean_on = lambda x, m: float(x[0, m].mean()) if bool(m.any()) else float("nan")
        stat.update(src_cos_inst=mean_on(cs, inst), src_cos_bg=mean_on(cs, ~inst))
        if src > 0:
            vs = torch.zeros_like(d)
            vs[:, inst] = _toward(dh[:, inst], s_dir[:, inst], -src * torch.asin(cs[:, inst].clamp(0.0, 1.0)))
            v += vs
            stat.update(rot_src=src, src_deg=_deg(vs, inst))
        if bg > 0:
            on = bg_w > 0
            vb = torch.zeros_like(d)
            vb[:, on] = _toward(dh[:, on], s_dir[:, on],
                                bg * bg_w[on][None, :, None] * torch.acos(cs[:, on].clamp(-1.0, 1.0)))
            v += vb
            stat.update(rot_bg=bg, bg_deg=_deg(vb, on), bg_cos=mean_on(cs, on), bg_tokens=int(on.sum()))

    if U is not None:  # instance k toward the write into its target words
        vi = torch.zeros_like(d)
        cos_in = []
        for k in range(K):
            uk, idx = U[:, k:k + 1], masks[k]
            if not bool(idx.any()) or float(uk.norm()) == 0.0:
                cos_in.append(float("nan"))
                continue
            dk = dh[:, idx]
            ur = uk.expand_as(dk)
            cu = (dk * ur).sum(-1, keepdim=True)
            cos_in.append(float(cu[0].mean()))
            if ins > 0:
                vi[:, idx] += _toward(dk, ur, ins * torch.acos(cu.clamp(-1.0, 1.0)))
        v += vi
        stat.update(rot_ins=ins, ins_deg=_deg(vi, inst), sem_cos_inst=cos_in)

    # one rotation by the summed tangent (exponential map), length kept
    vn = v.norm(dim=-1, keepdim=True)
    turn = vn.clamp(max=max_turn)
    new = norm * (torch.cos(turn) * dh + torch.sin(turn) * v / vn.clamp_min(1e-12)) + mean
    moved = vn[..., 0] > 0
    delta = torch.where(moved[..., None], new - D, torch.zeros_like(D))
    deg = turn[0, :, 0] * 180 / math.pi
    bg_on = bg_w > 0
    stat.update(turn_inst=float(deg[inst].mean()) if bool(inst.any()) else 0.0,
                turn_bg=float(deg[bg_on].mean()) if bool(bg_on.any()) else 0.0,
                capped=float((vn[0, :, 0] > max_turn).float().mean()),
                edit_rel=float(delta[0].norm(dim=-1).mean() / D[0].norm(dim=-1).mean().clamp_min(1e-12)))
    return (ho + delta).to(h_out.dtype), stat


@dataclass
class RotationSteer:
    """Strengths are mice's final run (4B_ROT_mice_2). Steps: None = every step. ctc_steps (0, 1) is mice's: klein's
    steps 0-1 take sigma 1.0 -> 0.908, Turbo's 1.0 -> 0.905."""
    rot_ctc: float = 0.9
    rot_src: float = 0.25
    rot_bg: float = 0.35
    rot_ins: float = 0.2
    ctc_steps: tuple | None = (0, 1)
    src_steps: tuple | None = None
    bg_steps: tuple | None = None
    ins_steps: tuple | None = None
    sigma_terms: tuple = ("ctc", "src", "bg")
    blocks: set | None = None  # block names ("double_3", "single_7") steered; None = every block
    ins_blocks: set | None = None  # blocks with the instruction term (within `blocks`); None = every block
    ins_centre: str | None = None  # None = raw u_k; "prompts" / "text": minus the mean text update (see top)
    local_tau: float = 2.0
    bg_margin: float = 1.0
    bg_tau: float = 2.0
    max_turn_deg: float = 90.0
    stats: list = field(default_factory=list)

    def setup(self, L: Layout, scheduler):
        """Per sample. scheduler: the pipeline's, whose sigmas (set before the first forward) give each step's
        sigma."""
        self.L, self.scheduler = L, scheduler
        HW = L.H * L.W
        masks = torch.zeros(len(L.inst), HW, dtype=torch.bool)
        for k, idx in enumerate(L.inst):
            masks[k, idx] = True
        self.masks = masks
        self.pair_w = pair_weights(masks, L.H, L.W, self.local_tau)
        self.bg_w = bg_weights(masks, L.H, L.W, self.bg_margin, self.bg_tau)
        self._dev = {}
        self.stats, self.calls, self.step, self.sigma = [], 0, None, None

    def _tensors(self, device):
        if device not in self._dev:
            self._dev[device] = (self.masks.to(device), [w.to(device) for w in self.pair_w], self.bg_w.to(device),
                                 [w.to(device) for w in self.L.tgt_words])
        return self._dev[device]

    def begin(self, t: float):
        """Before every transformer forward (one per step: guidance 1). t = the timestep the model gets (sigma, in
        the latents' dtype); the step's sigma is read from the scheduler at full precision."""
        self.step = self.calls
        self.calls += 1
        sigmas = [float(s) for s in self.scheduler.sigmas]
        if self.step >= len(sigmas) - 1 or abs(t - sigmas[self.step]) > 0.01:
            raise RuntimeError(f"forward {self.step} at t = {t:.4f}, expected the sigmas {sigmas}: one forward per "
                               f"step only (no CFG)")
        self.sigma = sigmas[self.step]

    def strengths(self, name: str) -> dict:
        if self.blocks is not None and name not in self.blocks:
            return dict(ctc=0.0, src=0.0, bg=0.0, ins=0.0)
        on = lambda steps: steps is None or self.step in steps
        f = lambda t: self.sigma if t in self.sigma_terms else 1.0
        return dict(ctc=self.rot_ctc * f("ctc") if on(self.ctc_steps) else 0.0,
                    src=self.rot_src * f("src") if on(self.src_steps) else 0.0,
                    bg=self.rot_bg * f("bg") if on(self.bg_steps) else 0.0,
                    ins=self.rot_ins * f("ins") if on(self.ins_steps) and (self.ins_blocks is None or
                                                                           name in self.ins_blocks) else 0.0)

    def _instruction(self, Dt: torch.Tensor, words: list) -> torch.Tensor:
        """u_k from the text stream's update Dt [B, n_text, D]: raw, or minus the mean over the prompts / the text.
        Logs the mean cosine between the raw and the centred anchors (ins_raw_cos)."""
        if self.ins_centre is None:
            return instruction_directions(Dt, words)
        if self.ins_centre not in ("prompts", "text"):
            raise ValueError(f"ins_centre {self.ins_centre!r}: None, 'prompts' or 'text'")
        Dt = Dt.float()
        start = self.L.n_vis if self.ins_centre == "prompts" else 0
        U = instruction_directions(Dt - Dt[:, start:self.L.n_text].mean(dim=1, keepdim=True), words)
        raw = instruction_directions(Dt, words)
        on = [k for k, w in enumerate(words) if len(w)]
        self._raw_cos = float((U[:, on] * raw[:, on]).sum(-1).mean()) if on else float("nan")
        return U

    def block_hook(self, name: str):
        def hook(module, args, kwargs, out):
            txt_in, img_in = kwargs["encoder_hidden_states"], kwargs["hidden_states"]
            txt_out, img_out = out
            L, HW = self.L, self.L.H * self.L.W
            if img_in.shape[1] != 2 * HW or txt_in.shape[1] != L.n_text:
                raise ValueError(f"{name}: {txt_in.shape[1]} text + {img_in.shape[1]} image tokens, layout "
                                 f"{L.n_text} + {2 * HW}")
            st = self.strengths(name)
            if not any(v > 0 for v in st.values()) or not len(L.inst):
                return out
            masks, pair_w, bg_w, words = self._tensors(img_out.device)
            s_dir = source_direction(img_out[:, HW:] - img_in[:, HW:]) if st["src"] > 0 or st["bg"] > 0 else None
            U = self._instruction(txt_out - txt_in, words) if st["ins"] > 0 else None
            new, stat = rotate(img_in[:, :HW], img_out[:, :HW], masks, pair_w, bg_w, s_dir, U, st["ctc"],
                               st["src"], st["bg"], st["ins"], self.max_turn_deg * math.pi / 180)
            if U is not None and self.ins_centre is not None:
                stat["ins_raw_cos"] = self._raw_cos
            self.stats.append(dict(step=self.step, layer=name, sigma=self.sigma, **stat))
            return txt_out, torch.cat([new, img_out[:, HW:]], dim=1)
        return hook


@contextmanager
def steering(transformer, steer: RotationSteer, L: Layout, scheduler):
    """Attach steer to every block for one pipeline call."""
    steer.setup(L, scheduler)

    def pre(module, args, kwargs):
        t = kwargs["timestep"]
        if t.shape[0] != 1:
            raise ValueError("steering runs one sample at a time")
        steer.begin(float(t[0]))

    handles = [transformer.register_forward_pre_hook(pre, with_kwargs=True)]
    handles += [blk.register_forward_hook(steer.block_hook(f"double_{b}"), with_kwargs=True)
                for b, blk in enumerate(transformer.transformer_blocks)]
    handles += [blk.register_forward_hook(steer.block_hook(f"single_{b}"), with_kwargs=True)
                for b, blk in enumerate(transformer.single_transformer_blocks)]
    try:
        yield steer
    finally:
        for h in handles:
            h.remove()
        steer._dev = {}
