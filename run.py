# main.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
)

from pipelines.pipeline_tag_stablediffusion   import StableDiffusionTangentialDecomposedPipeline
from pipelines.pipeline_tag_stablediffusionXL import StableDiffusionXLTangentialDecomposedPipeline
from pipelines.pipeline_tag_stablediffusion3  import StableDiffusion3TangentialDecomposedPipeline

from utils.logging_utils import setup_logging, print_versions
from utils.ops_utils import (
    GenerationCfg, device_string, count_params, generate, save_image
)

SchedulerName = Literal["ddim", "pndm", "dpm_solver", "euler"]
SDVersion     = Literal["1.5", "2.1", "3", "XL"] 

def move_to_device(pipe, dev: str, enable_xformers: bool = False):
    pipe = pipe.to(dev)
    if enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers memory-efficient attention enabled.")
        except Exception as e:
            print(f"xFormers enable failed: {e}")
    return pipe

@dataclass
class PipelineCfg:
    sd_version: SDVersion = "2.1"
    scheduler: Optional[SchedulerName] = None
    enable_xformers: bool = False

@dataclass
class IOCfg:
    root: str = ""
    subdir_tgs_disabled: str = "without_TAG"  
    subdir_tgs_enabled: str = "with_TAG"      
    filename_style: Literal["legacy", "smart"] = "smart"

@dataclass
class AppCfg:
    generation: GenerationCfg = GenerationCfg()
    pipeline: PipelineCfg = PipelineCfg()
    io: IOCfg = IOCfg()
    keep_cwd: bool = True
    log_level: Literal["INFO", "DEBUG", "WARNING"] = "INFO"


def create_pipeline(cfg: PipelineCfg):
    sv = cfg.sd_version
    if sv == "2.1":
        return StableDiffusionTangentialDecomposedPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1"
        )
    if sv == "1.5":
        return StableDiffusionTangentialDecomposedPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            use_safetensors=True,
            safety_checker=None,
        )
    if sv == "XL":
        return StableDiffusionXLTangentialDecomposedPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            use_safetensors=True,
        )
    if sv == "3":
        return StableDiffusion3TangentialDecomposedPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            text_encoder_3=None, tokenizer_3=None,
            use_safetensors=True,
        )
    raise ValueError(f"Unsupported sd_version: {sv}")

def apply_scheduler(name: Optional[SchedulerName], pipe) -> None:
    if not name:
        return
    n = name.lower()
    if n == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif n == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif n in {"dpm_solver", "dpmsolver"}:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
    elif n == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

def default_size(sd_version: SDVersion) -> Tuple[int, int]:
    """
    sd_version별 기본 해상도 반환(정보 목적으로만).
    """
    return {
        "1.5": (512, 512),
        "2.1": (768, 768),
        "3": (1024, 1024),
        "XL": (1024, 1024),
    }[sd_version]

def _sanitize_filename(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"[^\w\-\.\+_]", "", text)

def build_smart_filename(gen: GenerationCfg, sd_version: str) -> str:
    parts = [f"SD{sd_version}"]
    if gen.prompt:
        parts.append(_sanitize_filename(gen.prompt))
    parts += [f"cfg{gen.cfg_scale}", f"tgs{gen.t_lr}-{gen.r_lr}", f"seed{gen.seed}"]
    return "_".join(parts) + ".png"

def resolve_save_path(
    gen: GenerationCfg,
    sd_version: str,
    *,
    filename_style: Literal["legacy", "smart"] = "smart",
    root: str = "",
    subdir_when_tgs_disabled: str = "without_TAG",
    subdir_when_tgs_enabled: str = "with_TAG",
) -> Path:
    """
    - root == "" for current working directory
    - (t_lr, r_lr) == (1.0, 1.0) → 
    - filename_style:
        - "legacy"  
        - "smart"  
    """
    base = Path(root).expanduser().resolve() if root.strip() else Path.cwd()
    sub = subdir_when_tgs_disabled if (gen.t_lr == 1.0 and gen.r_lr == 1.0) else subdir_when_tgs_enabled
    out_dir = base / sub
    fname = "legacy.png" if filename_style == "legacy" else build_smart_filename(gen, sd_version)
    return out_dir / fname


@dataclass
class _HydraCfgView: 
    generation: GenerationCfg
    pipeline: PipelineCfg
    io: IOCfg
    keep_cwd: bool
    log_level: str

@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg: AppCfg) -> None:
    if cfg.keep_cwd:
        os.chdir(hydra.utils.get_original_cwd())

    setup_logging(cfg.log_level)
    print_versions()
    print("=== Config ===\n" + OmegaConf.to_yaml(OmegaConf.structured(_HydraCfgView(
        generation=cfg.generation, pipeline=cfg.pipeline, io=cfg.io,
        keep_cwd=cfg.keep_cwd, log_level=cfg.log_level
    ))))

    dev = device_string()
    pipe = create_pipeline(cfg.pipeline)
    pipe = move_to_device(pipe, dev, enable_xformers=cfg.pipeline.enable_xformers)
    apply_scheduler(cfg.pipeline.scheduler, pipe)

    params = count_params(pipe)
    print(f"✅ Ready on {dev} | params={params/1e6:.2f}M | scheduler={pipe.scheduler.config._class_name}")
    w, h = default_size(cfg.pipeline.sd_version)
    print(f"Default size for SD{cfg.pipeline.sd_version}: ({w}, {h})")

    img = generate(pipe, cfg.generation, dev)

    dest_path = resolve_save_path(
        gen=cfg.generation,
        sd_version=cfg.pipeline.sd_version,
        filename_style=cfg.io.filename_style,
        root=cfg.io.root,
        subdir_when_tgs_disabled=cfg.io.subdir_tgs_disabled,
        subdir_when_tgs_enabled=cfg.io.subdir_tgs_enabled,
    )
    save_image(img, dest_path)  

if __name__ == "__main__":
    run()
