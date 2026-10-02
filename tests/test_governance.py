"""Module duty: Test governance.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import pandas as pd
from services.governance import (
    ScientificDataLeakageGate, DatasetCardBuilder, StatisticalUncertainty,
    DataDiff, HumanApprovalEvidence, ModelPromotionRegistry, EvidenceDAG,
    AgentEvaluation, ControlledAnalysisExtensionAPI, AnalysisExtension,
)
from core.tableau_views import TableauSheet, TableauDashboard

def sample():
    """Perform the sample operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return pd.DataFrame({"Region":["A","B","A","B"],"Category":["X","X","Y","Y"],"Sales":[10,20,30,40],"Profit":[1,2,3,4]})

def test_governance_objects():
    """Perform the test governance objects operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df=sample(); gate=ScientificDataLeakageGate().evaluate(df,"Profit")
    assert gate["status"] in {"pass","review","blocked"}
    assert DatasetCardBuilder.build(df)["rows"]==4
    u=StatisticalUncertainty.bootstrap_mean(df["Sales"],n_boot=20)
    assert u["lower"] <= u["estimate"] <= u["upper"]
    diff=DataDiff.compare(df,df.iloc[:3])
    assert diff["row_delta"]==-1

def test_promotion_requires_approval():
    """Perform the test promotion requires approval operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    r=ModelPromotionRegistry(); r.register("m1",{"r2":.8})
    a=HumanApprovalEvidence.create("promotion","approved","ok")
    assert r.promote("m1","candidate",a)["stage"]=="candidate"

def test_evidence_and_extensions():
    """Perform the test evidence and extensions operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    dag=EvidenceDAG(); a=HumanApprovalEvidence.create("step","approved","ok"); dag.add(a); assert a["evidence_id"] in dag.nodes
    api=ControlledAnalysisExtensionAPI(); api.register(AnalysisExtension("demo","1.0",["analysis"],lambda x:x+1,{"x":"number"})); assert api.execute("demo","analysis",x=2)==3
    assert AgentEvaluation.evaluate({"approved_steps":["a"]})["approval_count"]==1

def test_tableau_sheet_dashboard():
    """Perform the test tableau sheet dashboard operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df=sample(); s=TableauSheet("Sales",df,rows=["Region"],columns=["Sales"],marks={"color_col":None,"size_col":None,"text_col":None,"detail_col":None,"tooltip_col":None})
    assert s.render()["plan"].orientation=="horizontal"
    d=TableauDashboard("D",[s]); d.apply_filter("Region","A"); assert len(d.filtered_sheets()[0].dataframe)==2
