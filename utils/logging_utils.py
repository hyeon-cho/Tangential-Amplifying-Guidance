import logging
import torch

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

def print_versions() -> None:
    import diffusers, accelerate, transformers, sentencepiece  # noqa
    logging.info(
        "torch=%s | diffusers=%s | accelerate=%s | transformers=%s",
        torch.__version__, diffusers.__version__, accelerate.__version__, transformers.__version__
    )
