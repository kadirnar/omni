"""Inference: frame-synchronous generators and the chat CLI (omni.infer.chat)."""

from .duplex import DuplexGenerator, DuplexStep
from .generate import GenResult, OmniGenerator, sample_logits

__all__ = ["DuplexGenerator", "DuplexStep", "GenResult", "OmniGenerator", "sample_logits"]
