"""
LongCat-Image-Edit vendored from diffusers v0.41.0 (see the header of each file), and the loader every script uses.
"""
import torch

from .pipeline_longcat_image_edit import LongCatImageEditPipeline
from .transformer_longcat_image import LongCatImageTransformer2DModel

MODEL_ID = "meituan-longcat/LongCat-Image-Edit-Turbo"
BASE_ID = "meituan-longcat/LongCat-Image-Edit"
# (num_inference_steps, guidance_scale) of each model card's example
SAMPLING = {MODEL_ID: (8, 1.0), BASE_ID: (50, 4.5)}


def load_pipeline(model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.bfloat16,
                  cpu_offload: bool = False) -> LongCatImageEditPipeline:
    """The vendored pipeline with the vendored transformer.

    from_pretrained alone would build diffusers' own transformer class from model_index.json, so processors and
    block hooks installed on the vendored class would silently not run. Load the transformer here and pass it in.
    """
    transformer = LongCatImageTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
    pipe = LongCatImageEditPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=dtype)
    assert type(pipe.transformer).__module__ == LongCatImageTransformer2DModel.__module__, type(pipe.transformer)
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe
