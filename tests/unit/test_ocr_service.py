"""OCRService tests using pytesseract mock."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.services.ocr_service import OCRService, _preprocess_for_ocr


def test_is_available_bool() -> None:
    assert isinstance(OCRService.is_available(), bool)


def test_extract_text_returns_empty_when_tesseract_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (10, 10)).save(img_path)
    assert OCRService().extract_text(img_path) == ""


def test_extract_text_happy_path_with_stubbed_pytesseract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")
    fake_module = types.ModuleType("pytesseract")
    setattr(fake_module, "image_to_string", lambda img, lang=None: "Hello world\n")  # noqa: B010
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    img_path = tmp_path / "text.png"
    Image.new("RGB", (16, 16)).save(img_path)
    assert OCRService().extract_text(img_path) == "Hello world"


def test_extract_text_swallows_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")

    def _boom(img, lang=None) -> str:
        raise RuntimeError("tesseract exploded")

    fake_module = types.ModuleType("pytesseract")
    setattr(fake_module, "image_to_string", _boom)  # noqa: B010
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    img_path = tmp_path / "bad.png"
    Image.new("RGB", (16, 16)).save(img_path)
    assert OCRService().extract_text(img_path) == ""


def _faint_gray_on_white(path: Path, *, size: tuple[int, int] = (40, 40)) -> None:
    """Save a low-contrast image: near-white bg with a faint gray bar."""
    arr = np.full((size[1], size[0]), 250, dtype=np.uint8)  # near-white
    arr[10:30, 5:35] = 235  # faint gray block (delta 15)
    Image.fromarray(arr, mode="L").convert("RGB").save(path)


def test_preprocess_upscales_2x_and_raises_contrast(tmp_path: Path) -> None:
    """The preprocessing helper must double dimensions AND lift contrast."""
    img_path = tmp_path / "faint.png"
    _faint_gray_on_white(img_path)

    raw = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    out = _preprocess_for_ocr(img_path)

    # 2x upscale on both axes.
    assert out.shape[0] == raw.shape[0] * 2
    assert out.shape[1] == raw.shape[1] * 2
    # Otsu binarisation lifts faint gray-on-white contrast: pixel std-dev
    # (spread between text and background) increases.
    assert float(out.std()) > float(raw.std())


def test_preprocess_raises_valueerror_on_unreadable(tmp_path: Path) -> None:
    """Unreadable path surfaces as ValueError so extract_text can swallow it."""
    with pytest.raises(ValueError, match="could not read"):
        _preprocess_for_ocr(tmp_path / "does-not-exist.png")


def test_extract_text_union_adds_preprocessed_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preprocessed pass contributes text the raw pass misses (degrade-only)."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")

    # Raw pass (PIL image) reads nothing; preprocessed pass (ndarray) reads text.
    def _image_to_string(img, lang=None) -> str:
        return "Add Todo" if isinstance(img, np.ndarray) else ""

    fake_module = types.ModuleType("pytesseract")
    setattr(fake_module, "image_to_string", _image_to_string)  # noqa: B010
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    img_path = tmp_path / "faint.png"
    _faint_gray_on_white(img_path)
    assert OCRService().extract_text(img_path) == "Add Todo"


def test_extract_text_joins_distinct_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Distinct, non-overlapping text from each pass is unioned, never dropped."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")

    def _image_to_string(img, lang=None) -> str:
        return "sharp header" if isinstance(img, np.ndarray) else "faint label"

    fake_module = types.ModuleType("pytesseract")
    setattr(fake_module, "image_to_string", _image_to_string)  # noqa: B010
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    img_path = tmp_path / "faint.png"
    _faint_gray_on_white(img_path)
    result = OCRService().extract_text(img_path)
    # Both fragments survive; substring matching downstream finds either.
    assert "faint label" in result
    assert "sharp header" in result


def test_extract_text_preprocess_failure_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken preprocessing pass never reduces the raw result."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")

    fake_module = types.ModuleType("pytesseract")
    setattr(fake_module, "image_to_string", lambda img, lang=None: "raw text")  # noqa: B010
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    def _boom(_: Path) -> np.ndarray:
        raise RuntimeError("cv2 blew up")

    monkeypatch.setattr("src.services.ocr_service._preprocess_for_ocr", _boom)

    img_path = tmp_path / "text.png"
    Image.new("RGB", (16, 16)).save(img_path)
    assert OCRService().extract_text(img_path) == "raw text"
