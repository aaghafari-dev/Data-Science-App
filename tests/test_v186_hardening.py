"""Module duty: Test v186 hardening.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from services.compute_backend import ComputeBackend
from services.agent_tools import default_registry

def test_compute_info_allows_missing_vram_without_formatting_failure():
    """Perform the test compute info allows missing vram without formatting failure operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    info=ComputeBackend.resolve("CPU+GPU", runtime_validate=False)
    assert info.requested_mode=="CPU+GPU"
    assert info.message

def test_uncertainty_and_robustness_tools_are_registered():
    """Perform the test uncertainty and robustness tools are registered operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    names={x["name"] for x in default_registry().describe()}
    assert "prediction_uncertainty" in names
    assert "robustness_perturbation" in names
