"""Tesseract-backed OCR for extracted frames."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _preprocess_for_ocr(image_path: Path) -> np.ndarray:
    """Return a contrast-enhanced, 2x-upscaled grayscale variant for OCR.

    Lifts faint gray-on-white UI text (e.g. low-contrast TodoMVC frames) so
    tesseract can read it. Pure enhancement: grayscale -> Otsu binarisation
    (separates faint gray text from a near-white background into solid
    black-on-white) -> 2x INTER_CUBIC upscale. Returns a numpy ndarray that
    ``pytesseract.image_to_string`` accepts directly. The result is only ever
    UNIONed with the raw pass downstream, so an unhelpful threshold on an
    already-good frame can never subtract a match.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"cv2 could not read image: {image_path}"
        raise ValueError(msg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _threshold, enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = enhanced.shape[:2]
    return cv2.resize(enhanced, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)


class OCRService:
    """Thin wrapper around pytesseract with graceful fallback."""

    def __init__(self, *, language: str = "eng") -> None:
        self._language = language

    @staticmethod
    def is_available() -> bool:
        """Return True when the ``tesseract`` binary is on PATH."""
        return shutil.which("tesseract") is not None

    def extract_text(self, image_path: Path) -> str:
        """Extract textual content from ``image_path`` using tesseract."""
        if not self.is_available():
            return ""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("pytesseract/Pillow not installed; OCR disabled")
            return ""
        try:
            with Image.open(image_path) as img:
                text: Any = pytesseract.image_to_string(img, lang=self._language)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("OCR failed for %s: %s", image_path, exc)
            return ""
        raw = str(text or "").strip()

        # Second pass on a contrast-enhanced variant. Downstream matching is
        # pure substring containment, so returning the UNION of raw +
        # preprocessed can only ADD matches, never remove one — degrade-only
        # by construction. Any failure here falls back to the raw result;
        # preprocessing must NEVER hard-fail or reduce output.
        try:
            preprocessed = _preprocess_for_ocr(image_path)
            enhanced_text: Any = pytesseract.image_to_string(preprocessed, lang=self._language)
            extra = str(enhanced_text or "").strip()
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            logger.warning("OCR preprocessing pass failed for %s: %s", image_path, exc)
            return raw

        # Substring-dedup union: since downstream matching is containment,
        # text already present in the other pass adds nothing.
        if not extra or extra in raw:
            return raw
        if not raw or raw in extra:
            return extra
        return f"{raw}\n{extra}".strip()
