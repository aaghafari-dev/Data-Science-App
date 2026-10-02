"""Module duty: Sharing.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from pathlib import Path
import json, zipfile

class SharingService:
    @staticmethod
    def build_bundle(output_path, artifacts: dict[str, bytes | str]):
        """Perform the build bundle operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        output=Path(output_path)
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
            for name, value in artifacts.items():
                if isinstance(value, bytes): z.writestr(name,value)
                else: z.writestr(name,str(value).encode("utf-8"))
            z.writestr("SHARE_MANIFEST.json", json.dumps({"artifacts":list(artifacts)},indent=2).encode("utf-8"))
        return output
