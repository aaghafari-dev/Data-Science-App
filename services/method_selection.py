from __future__ import annotations

from typing import Any
import pandas as pd


class AnalyticalMethodAdvisor:
    """Question/data-structure driven statistical-method candidates with assumptions."""
    @staticmethod
    def advise(question: str, df: pd.DataFrame | None = None) -> dict[str, Any]:
        q=(question or "").lower()
        methods=[]
        if any(x in q for x in ("correlation","association","related")):
            methods=[{"method":"Pearson correlation","when":"approximately linear numeric association","assumptions":["numeric variables","linearity matters","outliers reviewed"]},{"method":"Spearman correlation","when":"monotonic association or ordinal/rank structure","assumptions":["independent observations","monotonic relationship"]}]
        elif any(x in q for x in ("two groups","difference between two","compare two")):
            methods=[{"method":"Welch t-test","when":"difference in means for two independent groups","assumptions":["independent observations","numeric outcome","distribution/outliers reviewed"]},{"method":"paired t-test","when":"paired/repeated measurements","assumptions":["meaningful pairs","difference distribution reviewed"]}]
        elif any(x in q for x in ("three groups","multiple groups","anova")):
            methods=[{"method":"ANOVA","when":"mean differences across multiple groups","assumptions":["independence","residual structure reviewed","variance structure reviewed"]},{"method":"Kruskal-Wallis","when":"rank-based alternative is appropriate","assumptions":["independent groups","ordinal/continuous outcome"]}]
        elif any(x in q for x in ("categorical","chi-square","proportion")):
            methods=[{"method":"Chi-square test","when":"association between categorical variables","assumptions":["independent observations","expected counts reviewed"]},{"method":"Fisher exact test","when":"small contingency-table counts","assumptions":["categorical counts"]}]
        elif any(x in q for x in ("predict","forecast","model")):
            methods=[{"method":"Regression / supervised learning","when":"prediction or adjusted association","assumptions":["target definition","validation strategy","leakage control"]}]
        else:
            methods=[{"method":"Descriptive + exploratory analysis","when":"question is not sufficiently specified","assumptions":["define estimand/target before inferential testing"]}]
        return {"status":"ok","question":question,"candidates":methods,"multiple_testing_note":"If multiple hypotheses are tested, use an explicit multiplicity-control policy appropriate to the study design.","human_review_required":True}
