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
python check_binding.py                  # CPU, no weights: binding rules, leakage per block type, positions
python check_binding.py --tokens         # CPU, tokenizer files only: the real layouts of bench samples
python check_binding.py --pipeline       # CPU: the pipeline's own __call__ with tiny random VAE/transformer
python check_steer.py                    # CPU: rotation == mice's ResidualRotate; hook slicing on a tiny model
python check_steer.py --pipeline         # CPU: steering through the pipeline's own __call__
```

In bf16 the model is about 29 GB: a 6.2B transformer, the Qwen2.5-VL-7B text encoder (8.3B) and the FLUX.1 VAE.
Without offload, a 48 GB GPU ran out of memory on the second bench image (2026-11-26), so pass `--cpu_offload` there:
the text encoder, transformer and VAE then take turns on the GPU (about 18 GB according to the model card).

## Run

```bash
./run_all.sh mice                          # end to end, binding, steered (tuned defaults), 2 GPUs
./run_all.sh both --sample_ids 06 09 276   # smoke test on both benches; NGPU=1 for one GPU, DRY=1 to print
python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name smoke --sample_ids 06 09 276 --cpu_offload
./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice --cpu_offload
./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/lomoe_bench --exp_name turbo_lomoe --cpu_offload
python infer_baseline.py --model meituan-longcat/LongCat-Image-Edit --exp_name base_mice --cpu_offload  # 50 steps, CFG 4.5
python infer_bind.py --sample_ids 06 09 276 --exp_name bind_smoke --cpu_offload
python infer_bind.py --sample_ids 06 09 276 --exp_name bind_smoke_stacked --prompt_pos stacked --cpu_offload
python infer_steer.py --sample_ids 06 09 276 --exp_name rot_smoke --cpu_offload
```

`--cpu_offload` is a run-control flag: it is not part of the folder's settings, so a run can resume with or without
it.

Results go to `results/longcat_<exp_name>/` (baseline), `results/longcatbind_<exp_name>/` (binding) and `results/longcatrot_<exp_name>/` (binding + steering) as `{id}_result.png`, resized to the source size as in mice. The binding run also writes `{id}_layout.json` (token counts, and `vision_diff`, which should be 0); the steering run also `{id}_steer.json`, the per-(step, block) turns (deg) and cosines to the anchors. Steps and
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
| `rosie/tokens.py` | Token assignment: the vision span, the instruction slots, target-word positions, grid masks, the joint-sequence `Layout` and its text positions. |
| `rosie/binding.py` | The binding rules (mice's final binding), the row classes they split into, and `BindAttnProcessor`. |
| `rosie/instance_edit.py` | One edit: wraps the pipeline's `encode_prompt` (global + instance prompts) and `prepare_latents` (image grid start), and installs the processor. |
| `infer_bind.py` | Binding only, no steering, on the bench. Guidance 1 (Turbo) only. |
| `check_binding.py` | CPU checks: tiny random model (rules, leakage, positions); `--tokens` with the real tokenizer and processor; `--pipeline`, the pipeline's own `__call__` with a tiny random VAE and transformer. |
| `rosie/steer.py` | Rotation steering: block hooks that turn each block's update of the target tokens (mice's `residual_rotate.py`, same math). |
| `infer_steer.py` | Binding + steering on the bench (mice's final strengths by default). |
| `check_steer.py` | CPU checks: the rotation equals mice's `ResidualRotate` (needs `../mice`), the hooks read the anchors at the layout's positions, each term acts; `--pipeline` through the pipeline's own `__call__`. |
| `show_tokens.py` | Each instance's token region (after strict carving) drawn on the source and on a run's results, to tell a token-attribution border from the model's. |
| `steer_summary.py` | A steering run's `{id}_steer.json` per step and block kind (turns, cosine to the instruction anchor), optionally next to mice's run of the same sample. |
| `run_all.sh` | End to end, binding only, and binding + steering on one or both benches; the tuned steering (ctc 0.5, src 0.6, bg 0.35, ins 0.025, centred over the prompts) by default, every value overridable from the environment. |
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

1. Baseline (written; CPU checks pass and `check_vendored.py --full` passed on GPU, 2026-11-26): Turbo on both
   benches, with `--cpu_offload` on 48 GB cards.
2. Token assignment (written 2026-11-27, CPU-checked): every prompt is encoded the pipeline's way, with the source
   image (the only encoding this edit model was trained on). The encoder is causal and the vision span comes first,
   so the vision span is the same in every encoding (the stub check and `vision_diff` confirm this). It is kept
   once, and each prompt keeps its whole 512-token slot, padding included:
   `[vision span | global slot " "] [slot 1] ... [slot K] [target] [source]`. Each instance adds 512 tokens: on MICE-Bench the
   median sample has 12.7k tokens against about 9.1k for the baseline, and the largest, 74 (30 instances), 24.6k.
   Samples 167, 255 and 259 have one mask more than prompts; mice has no result for them, and `infer_bind.py`
   skips them (`Bench.paired`).
   Text positions: `--prompt_pos shared` (default) puts every slot at the global slot's positions and the image
   where a one-prompt edit has it, so each token sees its prompt and the image as in an ordinary edit. `stacked`
   is what the pipeline does with a longer text: each slot adds 512 positions before the image. On Qwen-2.1 the
   stacked layout let only the first prompt through.
3. Binding (written 2026-11-27, CPU-checked, not yet run on GPU): `BindAttnProcessor` in place of
   `LongCatImageAttnProcessor`, with mice's final rules (`rosie/binding.py`). A prompt is its vision span plus its slot:
   an instance's target tokens read the vision span, their slot and every image token; the background reads the
   global stream and every image token; each slot reads the vision span, itself and its instance's image tokens.
   Image-to-image attention stays open. Attention runs per row class on gathered keys, so no (n, n) mask is ever
   built. With the binding off it equals the model's processor exactly.
4. Steering (written 2026-11-27, CPU-checked, not yet run on GPU; binding ran normally on GPU, better than on
   Qwen-2.1): mice's `ResidualRotate` math, rosie-qwen's port (equal to mice on random inputs). Defaults are the
   paper's Ours (4B_ROT_mice_2): contact 0.9 on steps 0-1 (klein's steps 0-1 take sigma 1.0 -> 0.908, Turbo's
   1.0 -> 0.905), source 0.25, background 0.35 and instruction 0.2 on every step, contact/source/background scaled by
   sigma, every block (10 double + 20 single; mice steered all 25 of klein's, instruction 0.2 in both kinds). Both
   block kinds take `(image = [target | source], text)` and return `(text, image)`, so one hook reads the source
   write from the image stream and the target-word write from the text stream of the same block. There is no
   cache, so the anchors are recomputed in every block and step, as in mice. Turbo has 8 steps where klein had 4, so
   the every-step terms act twice as often over the same noise range: the first thing to tune if the edits come
   out over-steered. `--ins_blocks double|single|<names>` limits the instruction term to some blocks;
   `--ins_centre prompts` turns toward the target words' update minus the mean update over every instruction slot
   (global and instances, padding included), `--ins_centre text` minus the mean over the whole text stream (vision
   span too). In the user's first GPU runs (2026-11-27) the raw instruction term broke the image and source turning
   helped.

License: Apache-2.0 (code and weights).
