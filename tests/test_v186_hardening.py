from services.compute_backend import ComputeBackend
from services.agent_tools import default_registry

def test_compute_info_allows_missing_vram_without_formatting_failure():
    info=ComputeBackend.resolve("CPU+GPU", runtime_validate=False)
    assert info.requested_mode=="CPU+GPU"
    assert info.message

def test_uncertainty_and_robustness_tools_are_registered():
    names={x["name"] for x in default_registry().describe()}
    assert "prediction_uncertainty" in names
    assert "robustness_perturbation" in names
