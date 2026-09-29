from __future__ import annotations

from pathlib import Path
import json, zipfile

class SharingService:
    @staticmethod
    def build_bundle(output_path, artifacts: dict[str, bytes | str]):
        output=Path(output_path)
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
            for name, value in artifacts.items():
                if isinstance(value, bytes): z.writestr(name,value)
                else: z.writestr(name,str(value).encode("utf-8"))
            z.writestr("SHARE_MANIFEST.json", json.dumps({"artifacts":list(artifacts)},indent=2).encode("utf-8"))
        return output
