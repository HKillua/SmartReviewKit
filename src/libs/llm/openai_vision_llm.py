"""OpenAI-compatible Vision LLM implementation.

Works with any provider exposing an OpenAI-compatible chat completions API
that supports image_url content (OpenAI, ZhiPu GLM-4V, etc.).
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class OpenAIVisionLLMError(RuntimeError):
    """Raised when OpenAI-compatible Vision API call fails."""


class OpenAIVisionLLM(BaseVisionLLM):
    """Vision LLM backed by any OpenAI-compatible API (OpenAI, ZhiPu, etc.).

    Uses the standard ``openai`` Python SDK so that swapping providers only
    requires changing ``base_url`` and ``api_key``.
    """

    DEFAULT_MAX_IMAGE_SIZE = 2048

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai>=1.0"
            ) from exc

        vision_cfg = getattr(settings, "vision_llm", None)

        resolved_api_key = (
            api_key
            or (getattr(vision_cfg, "api_key", None) if vision_cfg else None)
            or os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

        resolved_base_url = (
            base_url
            or (getattr(vision_cfg, "base_url", None) if vision_cfg else None)
            or None
        )

        self._model = (
            model
            or (getattr(vision_cfg, "model", None) if vision_cfg else None)
            or "glm-4v-flash"
        )

        self.max_image_size = (
            max_image_size
            or (getattr(vision_cfg, "max_image_size", None) if vision_cfg else None)
            or self.DEFAULT_MAX_IMAGE_SIZE
        )

        self.default_temperature = getattr(settings.llm, "temperature", 0.7)
        self.default_max_tokens = min(
            getattr(settings.llm, "max_tokens", 1024), 1024
        )

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.validate_text(text)
        self.validate_image(image)

        processed = self.preprocess_image(
            image, max_size=(self.max_image_size, self.max_image_size)
        )
        img_b64 = self._to_base64(processed)

        api_messages: list[dict[str, Any]] = []
        if messages:
            api_messages.extend(
                [{"role": m.role, "content": m.content} for m in messages]
            )

        api_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{processed.mime_type};base64,{img_b64}"
                        },
                    },
                ],
            }
        )

        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        try:
            # Client creation moved here to use instance config
            client_kwargs: dict[str, Any] = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            import openai
            client = openai.OpenAI(**client_kwargs)
            
            resp = client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            usage = resp.usage.model_dump() if resp.usage else None
            return ChatResponse(
                content=choice.message.content or "",
                model=resp.model or self._model,
                usage=usage,
                raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
            )
        except Exception as e:
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] API call failed: {type(e).__name__}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Image preprocessing (reuse from AzureVisionLLM pattern)
    # ------------------------------------------------------------------

    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        if not max_size:
            return image
        try:
            from PIL import Image
        except ImportError:
            return image

        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            return image
        else:
            return image

        img = Image.open(io.BytesIO(image_bytes))
        
        # Normalize image mode before saving to JPEG.
        # RGBA/LA/P-mode images can carry transparency, which Pillow cannot write
        # directly as JPEG. Composite them onto a white background first so PPTX/PDF
        # extracted slide images remain captionable instead of failing hard.
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.getchannel("A") if "A" in img.getbands() else None
            background.paste(img.convert("RGBA"), mask=alpha)
            img = background
        elif img.mode == "P" and "transparency" in img.info:
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            img = background
        elif img.format in ("GIF", "WMF") or img.mode != "RGB":
            img = img.convert("RGB")
            
        w, h = img.size
        max_w, max_h = max_size
        
        # If resizing is needed
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return ImageInput(data=buf.getvalue(), mime_type="image/jpeg")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_base64(image: ImageInput) -> str:
        if image.base64:
            return image.base64
        if image.data:
            return base64.b64encode(image.data).decode("utf-8")
        if image.path:
            return base64.b64encode(Path(image.path).read_bytes()).decode("utf-8")
        raise OpenAIVisionLLMError("ImageInput has no valid data source")
