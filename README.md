# rosie-longcat

RoSIE (rotation-based residual steering, developed on FLUX.2-klein-4B in `../mice`) ported to
[LongCat-Image-Edit-Turbo](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo) to test whether it
carries over to a second backbone. The non-distilled
[LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) runs through the same code.

## Setup

The pins are the same as `../rosie-qwen`'s (diffusers 0.41.0, transformers 5.x), so its venv works:
`source ../rosie-qwen/.venv/bin/activate`. A standalone environment works too: `uv venv .venv && uv pip install -r
requirements.txt`. Never install these into the mice venv.

```bash
python check_vendored.py                 # CPU, no weights: vendored == installed diffusers
python check_vendored.py --processor     # CPU, tokenizer files only: what the text stream holds
python check_vendored.py --full          # GPU: whole pipeline, official vs vendored, max pixel diff must be 0
```

In bf16 the model is about 29 GB: a 6.2B transformer, the Qwen2.5-VL-7B text encoder (8.3B) and the FLUX.1 VAE. One
48 GB GPU holds all of it. On smaller cards, pass `--cpu_offload` (about 18 GB according to the model card).

## Run

```bash
python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name smoke --sample_ids 06 09 276
./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice
./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/lomoe_bench --exp_name turbo_lomoe
python infer_baseline.py --model meituan-longcat/LongCat-Image-Edit --exp_name base_mice    # 50 steps, CFG 4.5
```

Results go to `results/longcat_<exp_name>/{id}_result.png`, resized to the source size as in mice. Steps and
guidance default to the model card's, and `args.json` records the values used. `stats_shard<k>.json` has the
seconds per image and the peak GPU memory.

## Layout

| Path | What |
|---|---|
| `longcat/transformer_longcat_image.py`, `longcat/pipeline_longcat_image_edit.py` | Copied from diffusers v0.41.0 with only the imports changed. **Never edit them**: experiments wrap them at runtime (attention processors, block hooks) or live in new files. |
| `longcat/__init__.py` | `load_pipeline()`: the vendored pipeline with the vendored transformer (plain `from_pretrained` would load diffusers' own class); model ids and the model cards' steps and guidance. |
| `bench.py` | MICE-Bench / LoMOE-Bench loader, and `output_size()`, the pipeline's generation size. |
| `infer_baseline.py` | The official edit with one instruction (`edit_inst_single`) and no instance control. |
| `check_vendored.py` | Equality checks for the vendored code, and the text-stream check. |
| `runs.py`, `run_2gpu.sh` | Run-folder conventions and the two-GPU launcher, as in `../rosie-qwen`. |

## What differs from FLUX.2-klein (read from the diffusers code)

- **Blocks**: 10 double-stream then 20 single-stream blocks, width 3072 (24 heads × 128), the FLUX.1 block design.
  Double blocks have separate text and image weights and return (text, image), with the image stream
  `[target | source]`. Single blocks take text and image separately, run on `[text | target | source]` and return the
  pair; klein's single blocks take one concatenated tensor.
- **Positions**: RoPE ids are (modality, h, w). Text token i is at (0, i, i), target token (r, c) at (1, L + r, L + c)
  and source token (r, c) at (2, L + r, L + c), with L the text length. The source is resized to the generation size,
  so source token i and target token i share (h, w). Text lies on the diagonal before the image grid, as in
  Qwen-Image-2.1; on klein, text had its own axis.
- **Size**: area 1024², fixed in the pipeline, sides rounded up to multiples of 16; one token covers 16 px. Sample 06
  (640×480) becomes 1184×896, a 56×74 grid.
- **Text stream**: Qwen2.5-VL-7B sees the source at half the generation size. The stream is the encoder's last hidden
  states from `<|vision_start|>` on, with the system prefix and the assistant suffix cut: the vision tokens, then the
  instruction, padded on the right to 512 tokens. The transformer gets no mask, so the padding is attended to. For
  sample 06: 850 tokens = 336 vision + 2 vision markers + 135 instruction + 377 padding. Quoted text (`'...'`,
  `"..."`) is tokenized character by character; no MICE-Bench or LoMOE-Bench instruction contains quotes.
- **Attention**: the transformer passes no attention arguments to its blocks, so a mask has to come from our own
  processor. There is no causal rule or cache: every step is a full pass over the joint sequence.
- **Sampling**: Turbo uses 8 steps at guidance 1, one forward per step, with σ = 1.0, .957, .905, .840, .760, .655,
  .513, .311 (a linspace with a fixed exponential shift, mu = 1.15). The base Edit model uses 50 steps at CFG 4.5,
  two forwards per step. `num_inference_steps` works here because there is no baked schedule.

## Plan

1. Baseline (written; CPU checks pass, not yet run on GPU): `check_vendored.py --full`, then Turbo on both benches.
2. Token assignment: grid masks, instance prompts and target-word spans in the text stream above. Open: whether
   each instance prompt is encoded with the source image.
3. Binding: our processor in place of `LongCatImageAttnProcessor`, with mice's rules on `[text | target | source]`.
4. Steering: mice's `ResidualRotate` hooks; the single-block hook adapts to the (text, image) pair.

License: Apache-2.0 (code and weights).
