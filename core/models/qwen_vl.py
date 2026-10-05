"""Qwen2.5-VL-3B wrapper for multiple-choice answering.

Images are downscaled to a 448px long edge before the processor sees them,
following the measurement recorded in AIC commit 2b45157 (half-size keyframes at
JPEG q70 were a large speed win with no accuracy loss). It also keeps the visual
token count inside the 12GB budget: ~20 images at full 1080p would not fit
alongside SigLIP2 and jina-clip-v2.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

MAX_SIDE = 448
MAX_IMAGES = 20


class QwenVL:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        max_images: int = MAX_IMAGES,
        max_side: int = MAX_SIDE,
    ):
        self.model_id = model_id
        self.device = device
        self.max_images = max_images
        self.max_side = max_side
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        # device_map is left unset and .to() is explicit so the 12GB budget stays
        # predictable instead of accelerate deciding to offload mid-run.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        )
        self.model.eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Qwen VL loaded: %s on %s (%s)", self.model_id, self.device, dtype)

    def close(self) -> None:
        import torch

        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare(self, paths: Sequence[Path]):
        """Load and downscale images. AutoProcessor accepts PIL directly, so no
        qwen-vl-utils is needed (it is not installable on this network)."""
        from PIL import Image

        images = []
        for path in list(paths)[: self.max_images]:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:
                logger.warning("skipping unreadable image %s: %s", path, exc)
                continue
            w, h = img.size
            scale = min(1.0, self.max_side / max(w, h))
            if scale < 1.0:
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.BILINEAR,
                )
            images.append(img)
        return images

    def answer(
        self,
        images: Sequence[Path],
        prompt: str,
        max_new_tokens: int = 8,
    ) -> str:
        """Run one generation and return the raw decoded text.

        max_new_tokens is tiny on purpose: the task is to emit one letter, and a
        short cap stops the model from arguing with itself into a worse answer.
        """
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("QwenVL.load() must be called first")

        pil_images = self._prepare(images)
        content = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=pil_images or None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
