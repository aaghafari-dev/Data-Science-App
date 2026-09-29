from __future__ import annotations

from pathlib import Path
import json, zipfile

class PublicationPackageBuilder:
    @staticmethod
    def build(output_zip, files: dict[str, str | bytes]):
        output = Path(output_zip)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for arcname, content in files.items():
                if isinstance(content, bytes): z.writestr(arcname, content)
                else: z.write(content, arcname)
        return output
