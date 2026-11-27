"""
MICE-Bench / LoMOE-Bench loader (same LoMOE.json format: mice_bench/ and lomoe_bench/ under 3D-bench/image).

Images and masks are returned at their original size: the pipeline resizes the source itself (output_size below),
and the masks are put on the latent grid later, when tokens are assigned to instances.
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def _quoted(line: str) -> list[str]:
    return re.findall(r'"([^"]*)"', line)


def output_size(width: int, height: int) -> tuple[int, int]:
    """(width, height) the pipeline generates at for a source of this size: area 1024^2 (fixed in the pipeline), the
    source's aspect ratio, sides rounded up to multiples of 16. The latent grid is this / 16."""
    from longcat.pipeline_longcat_image_edit import calculate_dimensions  # the pipeline's own function
    return calculate_dimensions(1024 * 1024, width * 1.0 / height)


class Bench:
    def __init__(self, root: str):
        self.root = Path(root)
        self.entries = json.loads((self.root / "LoMOE.json").read_text())
        self.ids = sorted(self.entries, key=int)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, sample_id: str) -> dict:
        e = self.entries[sample_id]
        image = Image.open(self.root / e["image_path"]).convert("RGB")
        masks = [np.asarray(Image.open(self.root / p).convert("L")) > 127 for p in _quoted(e["mask_path"])]
        return dict(
            sample_id=sample_id,
            image=image,
            masks=masks,  # list of [H, W] bool, original size
            sources=_quoted(e.get("source_prompt", "")),
            targets=_quoted(e["fg_prompt"]),
            edit_inst_single=e["edit_inst_single"],
        )

    def paired(self, sample_id: str) -> bool:
        """One mask per instance prompt. Three MICE-Bench samples (167, 255, 259) have one mask more than prompts;
        mice has no result for them (Ours has 259 of 262), so the instance-controlled runs skip them too."""
        e = self.entries[sample_id]
        return len(_quoted(e["mask_path"])) == len(_quoted(e["fg_prompt"])) == len(_quoted(e.get("source_prompt", "")))

    def shard(self, num_shards: int = 1, shard_id: int = 0) -> list[str]:
        return self.ids[shard_id::num_shards]
