"""Artifact generator (Jinja2) tests."""

from __future__ import annotations

from pathlib import Path

from src.services.artifact_generator import ArtifactGenerator


def test_walkthrough_render() -> None:
    gen = ArtifactGenerator()
    output = gen.render(
        "walkthrough.md.j2",
        {
            "title": "Demo",
            "duration_seconds": 12.34,
            "width": 1920,
            "height": 1080,
            "synthesis": "A short walkthrough",
            "scenes": [
                {"start": 0.0, "end": 4.0, "description": "Intro"},
                {"start": 4.0, "end": 12.34, "description": "Wrap-up"},
            ],
            "transcript": "Hello world",
        },
    )
    assert "# Walkthrough: Demo" in output
    assert "1920x1080" in output
    assert "A short walkthrough" in output
    assert "Intro" in output
    assert "Wrap-up" in output
    assert "Hello world" in output


def test_walkthrough_render_without_scenes_or_transcript() -> None:
    gen = ArtifactGenerator()
    output = gen.render(
        "walkthrough.md.j2",
        {
            "title": "Empty",
            "duration_seconds": 1.0,
            "width": 640,
            "height": 480,
            "synthesis": None,
            "scenes": [],
            "transcript": None,
        },
    )
    assert "No scene boundaries detected" in output
    assert "No transcript available" in output


def test_ui_inventory_render() -> None:
    gen = ArtifactGenerator()
    output = gen.render(
        "ui_inventory.md.j2",
        {
            "title": "Demo",
            "ui_elements": [
                {"label": "Login button", "first_seen": 3.0, "ocr_text": "Sign in"},
            ],
        },
    )
    assert "# UI Inventory: Demo" in output
    assert "Login button" in output
    assert "Sign in" in output


def test_ui_inventory_render_empty() -> None:
    gen = ArtifactGenerator()
    output = gen.render("ui_inventory.md.j2", {"title": "Demo", "ui_elements": []})
    assert "No UI elements extracted" in output


def test_frame_index_render() -> None:
    gen = ArtifactGenerator()
    output = gen.render(
        "frame_index.md.j2",
        {
            "title": "Frames",
            "frames": [
                {"index": 0, "timestamp": 0.0, "description": "intro", "ocr_text": None},
                {"index": 1, "timestamp": 3.0, "description": "middle", "ocr_text": "Hello"},
            ],
        },
    )
    assert "| 0 | 0.00s | intro | - |" in output
    assert "| 1 | 3.00s | middle | Hello |" in output


def test_write_creates_file(tmp_path: Path) -> None:
    gen = ArtifactGenerator()
    dest = tmp_path / "out/walkthrough.md"
    gen.write(
        template_name="walkthrough.md.j2",
        context={
            "title": "Written",
            "duration_seconds": 2.0,
            "width": 10,
            "height": 20,
            "synthesis": "s",
            "scenes": [],
            "transcript": "t",
        },
        output_path=dest,
    )
    assert dest.exists()
    assert "Written" in dest.read_text()
