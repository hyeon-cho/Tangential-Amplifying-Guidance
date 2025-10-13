from __future__ import annotations

import time
import re
import logging
from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from contextlib import contextmanager

import torch


@dataclass
class GenerationCfg:
    prompt: str = ""
    seed: int = 10838
    cfg_scale: float = 0.0
    t_lr: float = 1.0
    r_lr: float = 1.0
    n_steps: int = 50

@dataclass
class IOCfg:
    save_path: str = ""
    filename_style: Literal["legacy", "smart"] = "smart"


def device_string() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def sanitize_filename(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"[^\w\-\.\+_]", "", text)


@contextmanager
def timed_cuda(label: str, use_cuda: bool):
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
    logging.info("%s: %.6fs", label, elapsed)


def count_params(pipe) -> int:
    total = 0
    logging.info("Counting parameters...")
    for name in ("unet", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
        if hasattr(pipe, name):
            m = getattr(pipe, name)
            if isinstance(m, torch.nn.Module):
                n = sum(p.numel() for p in m.parameters())
                total += n
                logging.info("  - %-15s : %.2fM", name, n / 1e6)
    return total

def generate(pipe, gen: GenerationCfg, dev: str):
    g = torch.Generator(device=dev).manual_seed(gen.seed)
    label = "Image generation (CUDA)" if dev == "cuda" else "Image generation (CPU)"
    with timed_cuda(label, use_cuda=(dev == "cuda")):
        out = pipe(
            gen.prompt,
            num_inference_steps=gen.n_steps,
            generator=g,
            guidance_scale=gen.cfg_scale,
            t_guidance_scale=gen.t_lr,
            r_guidance_scale=gen.r_lr,
        )
    return out.images[0]

def save_image(img, dest: Path, create_dirs: bool = True) -> Path:
    """
    dest: 최종 저장 파일 경로 (파일명 포함)
    create_dirs: 상위 디렉토리를 필요 시 생성
    """
    if create_dirs:
        dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    logging.info("Saved: %s", dest)
    return dest
