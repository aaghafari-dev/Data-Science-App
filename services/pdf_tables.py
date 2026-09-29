from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

class PDFTableExtractor:
    """Extract tables page-by-page and preserve page/table identity."""
    @staticmethod
    def extract(path: str) -> list[dict[str, Any]]:
        results=[]; p=Path(path)
        # Prefer tabula-py because it is already part of the application stack.
        try:
            import tabula
            for page in range(1, 1001):
                try:
                    tables=tabula.read_pdf(str(p), pages=str(page), multiple_tables=True, lattice=False, stream=True, pandas_options={"dtype":str})
                except Exception:
                    break
                if not tables:
                    # Continue because PDFs may have blank pages.
                    continue
                for i,df in enumerate(tables,1):
                    if isinstance(df,pd.DataFrame) and not df.empty and df.shape[1] >= 1:
                        df=PDFTableExtractor._clean(df)
                        if not df.empty:
                            results.append({"name":f"{p.stem} — Page {page} — Table {i}","page":page,"table_index":i,"data":df})
            if results: return results
        except Exception as tabula_error:
            last_error=tabula_error
        else:
            last_error="Tabula found no tables."
        # Optional pure-Python fallback for environments without Java/tabula.
        try:
            import pdfplumber
            with pdfplumber.open(str(p)) as pdf:
                for page_no,page in enumerate(pdf.pages,1):
                    tables=page.extract_tables() or []
                    for i,raw in enumerate(tables,1):
                        if not raw: continue
                        width=max(len(r or []) for r in raw)
                        rows=[(r or [])+[None]*(width-len(r or [])) for r in raw]
                        header=rows[0]; data=rows[1:]
                        df=pd.DataFrame(data,columns=[str(x).strip() if x not in (None,"") else f"Column_{j+1}" for j,x in enumerate(header)])
                        df=PDFTableExtractor._clean(df)
                        if not df.empty: results.append({"name":f"{p.stem} — Page {page_no} — Table {i}","page":page_no,"table_index":i,"data":df})
            if results:return results
        except Exception as exc:
            raise RuntimeError(f"Could not extract PDF tables. Tabula: {last_error}; pdfplumber fallback: {exc}") from exc
        raise RuntimeError(f"No usable tables were detected in the PDF. {last_error}")

    @staticmethod
    def _clean(df):
        work=df.copy()
        work.columns=[str(c).strip() if str(c).strip() and not str(c).lower().startswith("unnamed") else f"Column_{i+1}" for i,c in enumerate(work.columns)]
        work=work.dropna(axis=1,how="all").dropna(axis=0,how="all").reset_index(drop=True)
        # Try numeric conversion without destroying mixed columns.
        for c in work.columns:
            converted=pd.to_numeric(work[c],errors="coerce")
            if len(work) and converted.notna().mean() >= .9: work[c]=converted
        return work
