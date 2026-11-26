"""
Run folders shared by the inference scripts (the mice conventions): <output_dir>/<prefix>_<exp_name>/, args.json,
env_<host>_shard<k>.json, {id}_result.png resized to the source size, finished samples skipped, and a folder never
mixes settings.
"""
import json
import platform
from pathlib import Path

import torch

RUN_CONTROL = {"sample_ids", "num_shards", "shard_id", "device", "cpu_offload", "overwrite"}


def add_run_args(p):
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--sample_ids", type=str, nargs="+", default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--overwrite", action="store_true")


def run_folder(args, prefix: str) -> Path:
    save_dir = Path(args.output_dir) / f"{prefix}_{args.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in RUN_CONTROL}
    args_path = save_dir / "args.json"
    if args_path.exists() and not args.overwrite:
        old = {k: v for k, v in json.loads(args_path.read_text()).items() if k not in RUN_CONTROL}
        # a setting the old folder does not have counts as None, so a flag added later with default None
        # (e.g. --steps) does not lock earlier folders
        diff = {k: (old.get(k), settings.get(k)) for k in old.keys() | settings.keys()
                if old.get(k) != settings.get(k)}
        if diff:
            raise SystemExit(f"{save_dir} was made with other settings {diff} (old, new): use another --exp_name, "
                             f"or --overwrite")
    args_path.write_text(json.dumps(settings, indent=2))
    return save_dir


def todo(bench, args, save_dir: Path) -> list:
    ids = bench.shard(args.num_shards, args.shard_id)
    if args.sample_ids:
        ids = [i for i in ids if i in set(args.sample_ids)]
    if not args.overwrite:
        ids = [i for i in ids if not (save_dir / f"{i}_result.png").exists()]
    print(f"{len(ids)} samples to run -> {save_dir}")
    return ids


def save_environment(save_dir: Path, args, model: str):
    import diffusers
    import transformers
    env = dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda, diffusers=diffusers.__version__, transformers=transformers.__version__,
               gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    try:  # the snapshot folder name is the model revision actually loaded
        from huggingface_hub import snapshot_download
        env["model_snapshot"] = snapshot_download(model, local_files_only=True)
    except Exception as e:
        env["model_snapshot"] = f"{model} (no hub snapshot: {type(e).__name__})"
    (save_dir / f"env_{env['host']}_shard{args.shard_id}.json").write_text(json.dumps(env, indent=2))


def save_result(image, source, path: Path):
    """RGB (the VAE is RGBA; the benches and metrics are RGB), resized to the source size as in mice."""
    image = image.convert("RGB")
    if image.size != source.size:
        from PIL import Image
        image = image.resize(source.size, Image.LANCZOS)
    image.save(path)
