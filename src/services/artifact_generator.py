"""Render Jinja2 artifact templates (walkthrough, UI inventory, frame index)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


_TEMPLATE_ROOT = Path(__file__).parent.parent / "templates"


class ArtifactGenerator:
    """Generate Markdown artifacts from pipeline output."""

    def __init__(self, template_root: Path | None = None) -> None:
        root = template_root or _TEMPLATE_ROOT
        self._env = Environment(
            loader=FileSystemLoader(str(root)),
            autoescape=select_autoescape(enabled_extensions=("html",)),
            keep_trailing_newline=True,
        )

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a single template to a string."""
        template = self._env.get_template(template_name)
        return template.render(**context)

    def write(self, *, template_name: str, context: dict[str, Any], output_path: Path) -> Path:
        """Render a template and write it to ``output_path``."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.render(template_name, context)
        output_path.write_text(rendered, encoding="utf-8")
        return output_path
