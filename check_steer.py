"""
Checks for the rotation steering (rosie/steer.py). CPU, no weights.

    python check_steer.py [--mice ../mice]
        1. the port: rotate() == mice's ResidualRotate._edit (flux2/residual_rotate.py, read from --mice; skipped if
           it is not there) on random updates, with each term alone and all four together
        2. a tiny random transformer (2 double + 2 single blocks) on check_binding.py's synthetic layout, binding on,
           3 steps:
           a. all strengths 0: the hooks leave the output unchanged
           b. in every block and step, the steered image stream == rotate() of the block's own output with the
              anchors read at the layout's joint positions (source = L.src, target words = L.tgt_words; the centred
              instruction anchors minus the mean over the slots or the whole text), the source part and the text
              stream untouched
           c. each term alone changes the output and turns its tokens; contact only on its steps; the sigma terms
              scaled by the step's sigma
    python check_steer.py --pipeline --sample_ids 06   the pipeline's own __call__ (check_binding.tiny_pipeline):
        strengths 0 == binding only (same pixels); steering on: one stats entry per step and block, at the
        scheduler's sigmas
"""
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from check_binding import H, W, img_ids, synthetic_layout, tiny_model
from rosie.binding import binding
from rosie.steer import (RotationSteer, bg_weights, pair_weights, rotate, source_direction,
                         steering)

STRENGTHS = dict(ctc=0.9, src=0.25, bg=0.35, ins=0.2)
OK = True


def report(name, cond, extra=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {extra}")


def check_port(mice: Path) -> None:
    if not (mice / "flux2" / "residual_rotate.py").exists():
        print(f"skip the port check: no {mice}/flux2/residual_rotate.py")
        return
    sys.path.insert(0, str(mice))
    from flux2.residual_rotate import ResidualRotate

    torch.manual_seed(0)
    h, w, Dm = 6, 7, 32
    masks = torch.zeros(3, h, w)
    masks[0, :3, :3] = 1
    masks[1, :3, 3:5] = 1  # touches instance 0
    masks[2, 4:, 4:] = 1
    flat = masks.reshape(3, -1) > 0.5
    h_in, h_out = torch.randn(1, h * w, Dm), torch.randn(1, h * w, Dm)
    D_src = torch.randn(1, h * w, Dm)
    U = F.normalize(torch.randn(1, 3, Dm), dim=-1)
    pw, bw = pair_weights(flat, h, w, 2.0), bg_weights(flat, h, w, 1.0, 2.0)
    for name, on in [(t, {t}) for t in STRENGTHS] + [("all four", set(STRENGTHS))]:
        st = {t: (v if t in on else 0.0) for t, v in STRENGTHS.items()}
        ref = ResidualRotate(rot_ctc=st["ctc"], rot_src=st["src"], rot_bg=st["bg"], rot_ins=st["ins"],
                             rot_ins_double=0.0, local_tau=2.0, bg_margin=1.0, bg_tau=2.0, sigma_terms=())
        ref.setup(list(masks), device="cpu")
        ref.sigma, ref.step_idx, ref.factor = 1.0, 0, 1.0
        want = ref._edit(h_in, h_out, "single_0", "lat", src=(D_src, D_src), sem=U)
        got, _ = rotate(h_in, h_out, flat, pw, bw, source_direction(D_src), U, st["ctc"], st["src"], st["bg"],
                        st["ins"], math.pi / 2)
        d = (got - want).abs().max().item()
        report(f"port == mice ResidualRotate ({name})", d < 1e-5 and (got - h_out).abs().max() > 0,
               f"max|diff|={d:.2g}")


SIGMAS = [0.9, 0.6, 0.3]


def run(model, L, latents, text, steer=None, capture=None):
    """3 forwards at SIGMAS (a fake scheduler), binding on; returns the outputs. capture: per block, the raw output
    (hooks registered before the steering's) and the steered one (after)."""
    sched = SimpleNamespace(sigmas=torch.tensor(SIGMAS + [0.0]))
    blocks = [(f"double_{i}", b) for i, b in enumerate(model.transformer_blocks)] + \
             [(f"single_{i}", b) for i, b in enumerate(model.single_transformer_blocks)]
    handles = []

    def grab(name, key):
        def hook(module, args, kwargs, out):
            capture.setdefault(name, {}).setdefault(key, []).append(
                (kwargs["encoder_hidden_states"].clone(), kwargs["hidden_states"].clone(),
                 out[0].clone(), out[1].clone()))
        return hook

    if capture is not None:
        handles += [b.register_forward_hook(grab(n, "raw"), with_kwargs=True) for n, b in blocks]
    outs = []
    try:
        with binding(model, L):
            ctx = steering(model, steer, L, sched) if steer is not None else None
            if ctx is not None:
                ctx.__enter__()
            if capture is not None:
                handles += [b.register_forward_hook(grab(n, "steered"), with_kwargs=True) for n, b in blocks]
            try:
                for t in SIGMAS:
                    outs.append(model(hidden_states=latents, encoder_hidden_states=text, txt_ids=L.text_ids(),
                                      img_ids=img_ids(L), timestep=torch.tensor([t]), return_dict=False)[0])
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
    finally:
        for h in handles:
            h.remove()
    return outs


def check_anchors(model, L, latents, text, ref, masks, pw, bw, every, centre):
    """In every block and step the steered target == rotate() of the block's raw output, with the anchors read at the
    layout's joint positions; for a centred instruction anchor, the mean over the slots ([n_vis, n_text)) or the
    whole text ([0, n_text)) is subtracted from the target words' mean."""
    HW = H * W
    steer = RotationSteer(ins_centre=centre, **every)
    cap = {}
    out = run(model, L, latents, text, steer, cap)
    moved = max((a - b).abs().max().item() for a, b in zip(out, ref))
    worst, untouched, n = 0.0, True, 0
    for name, c in cap.items():
        for step, (raw, steered) in enumerate(zip(c["raw"], c["steered"])):
            t_in, i_in, t_out, i_out = raw
            joint_in, joint_out = torch.cat([t_in, i_in], 1), torch.cat([t_out, i_out], 1)
            D = (joint_out - joint_in).float()  # the block's update, read at the layout's joint positions
            sig = SIGMAS[step]
            st = dict(ctc=STRENGTHS["ctc"] * sig if step in (0, 1) else 0.0, src=STRENGTHS["src"] * sig,
                      bg=STRENGTHS["bg"] * sig, ins=STRENGTHS["ins"])
            U = torch.zeros(1, len(L.tgt_words), D.shape[-1])
            for k, words in enumerate(L.tgt_words):
                u = D[0, words].mean(0)
                if centre is not None:
                    u = u - D[0, (L.n_vis if centre == "prompts" else 0):L.n_text].mean(0)
                U[0, k] = F.normalize(u, dim=-1)
            want, _ = rotate(joint_in[:, L.tgt], joint_out[:, L.tgt], masks, pw, bw, source_direction(D[:, L.src]),
                             U, st["ctc"], st["src"], st["bg"], st["ins"], math.pi / 2)
            worst = max(worst, (steered[3][:, :HW] - want).abs().max().item())
            untouched &= torch.equal(steered[3][:, HW:], i_out[:, HW:]) and torch.equal(steered[2], t_out)
            n += 1
    report(f"[u_k {centre or 'raw'}] steered target == rotate() with anchors at the layout's positions "
           f"({n} block edits)", worst < 1e-5 and moved > 0, f"max|diff|={worst:.2g}, output moved {moved:.2g}")
    report(f"[u_k {centre or 'raw'}] source part and text stream untouched", untouched)
    if centre is None:
        steps_ctc = sorted({s["step"] for s in steer.stats if s.get("rot_ctc", 0) > 0})
        scaled = all(abs(s["rot_src"] - 0.25 * s["sigma"]) < 1e-9 and abs(s["rot_ins"] - 0.2) < 1e-9
                     for s in steer.stats)
        report("contact on steps 0, 1 only; src scaled by sigma, ins not", steps_ctc == [0, 1] and scaled,
               f"(contact steps {steps_ctc}, {len(steer.stats)} stats entries)")
    else:
        c = [s["ins_raw_cos"] for s in steer.stats]
        report(f"[u_k {centre}] cosine raw vs centred anchor logged", len(c) == n,
               f"(mean {sum(c) / len(c):.2f})")


def check_tiny():
    L = synthetic_layout()
    HW = H * W
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 2 * HW, 8, generator=g)
    text = torch.randn(1, L.n_text, 32, generator=g)
    every = dict(ctc_steps=(0, 1), src_steps=None, bg_steps=None, ins_steps=None, bg_margin=0.0, bg_tau=1.0)
    with torch.no_grad():
        model = tiny_model(2, 2)
        ref = run(model, L, latents, text)
        zero = RotationSteer(rot_ctc=0, rot_src=0, rot_bg=0, rot_ins=0, **every)
        d = max((a - b).abs().max().item() for a, b in zip(ref, run(model, L, latents, text, zero)))
        report("all strengths 0 == no steering", d == 0, f"max|diff|={d:.2g}")

        masks = torch.zeros(len(L.inst), HW, dtype=torch.bool)
        for k, idx in enumerate(L.inst):
            masks[k, idx] = True
        pw, bw = pair_weights(masks, H, W, 2.0), bg_weights(masks, H, W, every["bg_margin"], every["bg_tau"])
        for centre in (None, "prompts", "text"):
            check_anchors(model, L, latents, text, ref, masks, pw, bw, every, centre)

        for t in STRENGTHS:
            st = {f"rot_{u}": (v if u == t else 0.0) for u, v in STRENGTHS.items()}
            steer = RotationSteer(**st, **every)
            out = run(model, L, latents, text, steer)
            moved = max((a - b).abs().max().item() for a, b in zip(out, ref))
            deg = max(s.get(f"{t}_deg", 0.0) for s in steer.stats)
            report(f"{t} alone moves the output and turns its tokens", moved > 0 and deg > 0,
                   f"(output {moved:.2g}, up to {deg:.1f} deg)")


def check_pipeline(args):
    import numpy as np

    from bench import Bench
    from check_binding import tiny_pipeline
    from rosie.instance_edit import edit

    pipe = tiny_pipeline()
    s = Bench(args.dataset_root)[args.sample_ids[0]]
    steps = 3
    with torch.no_grad():
        bound, _ = edit(pipe, s, steps=steps)
        zero, _ = edit(pipe, s, steps=steps, steer=RotationSteer(rot_ctc=0, rot_src=0, rot_bg=0, rot_ins=0))
        report("strengths 0 == binding only (pixels)", np.array_equal(np.asarray(bound), np.asarray(zero)))
        steer = RotationSteer()
        out, prep = edit(pipe, s, steps=steps, steer=steer)
        sig = [float(x) for x in pipe.scheduler.sigmas[:steps]]
        n_blocks = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
        at = sorted({(e["step"], e["sigma"]) for e in steer.stats})
        report(f"steering on: {len(steer.stats)} stats entries = {steps} steps x {n_blocks} blocks, at the "
               f"scheduler's sigmas", len(steer.stats) == steps * n_blocks and at == list(enumerate(sig)) and
               not np.array_equal(np.asarray(out), np.asarray(bound)), f"{[round(x, 3) for x in sig]}")
        e = steer.stats[0]
        print(f"     step 0 {e['layer']}: turn_inst {e['turn_inst']:.1f} deg, turn_bg {e['turn_bg']:.1f} deg, "
              f"ins {e['ins_deg']:.1f}, ctc {e.get('ctc_deg', 0):.1f}, src {e['src_deg']:.1f}, bg {e['bg_deg']:.1f}")
        report("the hooks are gone afterwards", not any(b._forward_hooks for b in pipe.transformer.transformer_blocks)
               and not pipe.transformer._forward_pre_hooks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mice", type=Path, default=Path("../mice"), help="the mice repo, for the port check")
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_ids", type=str, nargs="+", default=["06"])
    args = p.parse_args()
    if args.pipeline:
        check_pipeline(args)
    else:
        check_port(args.mice)
        check_tiny()
    print("PASS" if OK else "FAIL")
    sys.exit(0 if OK else 1)
