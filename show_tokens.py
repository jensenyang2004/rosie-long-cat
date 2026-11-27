"""
Where each instance's tokens are: the source with every instance's token region (the binding and steering's, after
strict carving) filled in its colour and numbered, next to a result with the same regions outlined. A border in a
result that follows an outline is the token attribution; one that does not is the model.

    python show_tokens.py --sample_ids 06 09 276 --results results/longcatrot_rot_smoke
    python show_tokens.py --sample_ids 06 --out results/tokens          (source only)

Writes <out>/<id>_tokens.png (default out: <results>/tokens, or results/tokens). Prints, per instance, its tokens,
the tokens strict carving gave to smaller instances, and the target prompt. Needs no model.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bench import Bench, output_size
from rosie.tokens import grid_masks

COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180),
           (0, 0, 128), (128, 128, 128)]


def upsample(m: np.ndarray, size: tuple) -> np.ndarray:
    """Token grid [Hg, Wg] -> image size (w, h), each token a block (the pipeline stretches the source to the
    generation size, and results are resized back)."""
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def outline(m: np.ndarray, width: int = 2) -> np.ndarray:
    inner = m.copy()
    for _ in range(width):
        inner = inner & np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1) & np.roll(inner, -1, 1)
    return m & ~inner


def panel(image: Image.Image, regions: list, fill: bool) -> Image.Image:
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    for k, m in enumerate(regions):
        c = np.array(COLOURS[k % len(COLOURS)], np.float32)
        if fill:
            a[m] = 0.55 * a[m] + 0.45 * c
        a[outline(m)] = c
    out = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for k, m in enumerate(regions):
        if m.any():
            ys, xs = np.nonzero(m)
            x, y = int(xs.mean()), int(ys.mean())
            d.rectangle([x - 2, y - 2, x + 14, y + 12], fill=(0, 0, 0))
            d.text((x, y - 1), str(k), fill=COLOURS[k % len(COLOURS)])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_ids", type=str, nargs="+", required=True)
    p.add_argument("--results", type=str, default=None, help="a run folder with {id}_result.png")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no_strict", action="store_true")
    args = p.parse_args()
    out = Path(args.out or (Path(args.results) / "tokens" if args.results else "results/tokens"))
    out.mkdir(parents=True, exist_ok=True)
    bench = Bench(args.dataset_root)
    for sid in args.sample_ids:
        s = bench[sid]
        w, h = output_size(*s["image"].size)
        Hg, Wg = h // 16, w // 16
        strict = grid_masks(s["masks"], Hg, Wg, strict=not args.no_strict)
        loose = grid_masks(s["masks"], Hg, Wg, strict=False)
        size = s["image"].size
        regions = [upsample(m.numpy(), size) for m in strict]
        panels = [panel(s["image"], regions, fill=True)]
        if args.results:
            res = Path(args.results) / f"{sid}_result.png"
            if res.exists():
                panels.append(panel(Image.open(res), regions, fill=False))
            else:
                print(f"{sid}: no {res}")
        sheet = Image.new("RGB", (sum(q.size[0] for q in panels) + 8 * (len(panels) - 1), size[1]), (255, 255, 255))
        x = 0
        for q in panels:
            sheet.paste(q, (x, 0))
            x += q.size[0] + 8
        sheet.save(out / f"{sid}_tokens.png")
        print(f"{sid}: grid {Hg}x{Wg} -> {out / f'{sid}_tokens.png'}")
        for k, (a, b) in enumerate(zip(strict, loose)):
            lost = int(b.sum()) - int(a.sum())
            print(f"   {k:2d} {int(a.sum()):5d} tokens" + (f" ({lost} carved away)" if lost else "") +
                  f"  {s['sources'][k]!r} -> {s['targets'][k]!r}")


if __name__ == "__main__":
    main()
