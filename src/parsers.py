from __future__ import annotations

import os
from glob import glob
from typing import Any

import yaml


class OKFParser:
    @staticmethod
    def parse_file(file_path: str) -> dict[str, Any]:
        """Extract YAML frontmatter and body text from an OKF markdown file."""
        if not file_path or not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, encoding="utf-8") as handle:
                content = handle.read()
            parts = content.split("---")
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1]) or {}
                body = "---".join(parts[2:]).strip()
                return {"metadata": metadata, "body": body, "path": file_path}
        except (OSError, yaml.YAMLError) as exc:
            print(f"Error parsing OKF file {file_path}: {exc}")
        return {}

    @staticmethod
    def load_concept_files(bundle_dir: str) -> list[dict[str, Any]]:
        concepts_dir = os.path.join(bundle_dir, "concepts")
        paths = sorted(glob(os.path.join(concepts_dir, "*.md")))
        parsed = []
        for path in paths:
            data = OKFParser.parse_file(path)
            if data:
                parsed.append(data)
        return parsed
