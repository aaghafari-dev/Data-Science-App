"""Module duty: Publication package.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from pathlib import Path
import json, zipfile

class PublicationPackageBuilder:
    @staticmethod
    def build(output_zip, files: dict[str, str | bytes]):
        """Perform the build operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        output = Path(output_zip)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for arcname, content in files.items():
                if isinstance(content, bytes): z.writestr(arcname, content)
                else: z.write(content, arcname)
        return output
