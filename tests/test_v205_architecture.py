"""V20.5 architecture regression tests."""
import pandas as pd
from services.agent_provider_registry import AgentProviderRegistry
from services.llm_config import LLMConfig
from services.master_router import MasterDecisionEngine
from services.unsupervised_analysis import UnsupervisedAnalysisEngine
from services.reinforcement_learning import TabularRLEngine
from services.validation_strategy import ValidationStrategyAgent

def test_per_agent_provider_registry_isolated():
    reg=AgentProviderRegistry()
    reg.set("Agent Data Scientist", LLMConfig(provider="api",model_name="model-a",api_key="x"))
    reg.set("AI Agent Report", LLMConfig(provider="api",model_name="model-b",api_key="y"))
    assert reg.get("Agent Data Scientist").model_name=="model-a"
    assert reg.get("AI Agent Report").model_name=="model-b"

def test_master_routes_unsupervised_and_rl():
    df=pd.DataFrame({"x":[1,2,3,4,5,6],"y":[2,1,4,3,6,5]})
    u=MasterDecisionEngine.decide(df,"discover latent structure with unsupervised learning",None,target=None)
    r=MasterDecisionEngine.decide(df,"solve a sequential decision problem using reinforcement learning",None,target=None)
    assert u["primary_route"]=="UNSUPERVISED"
    assert "Unsupervised Learning Agent" in u["sub_agents"]
    assert r["primary_route"]=="REINFORCEMENT_LEARNING"
    assert "Reinforcement Learning Agent" in r["sub_agents"]

def test_validation_strategy_is_task_specific():
    df=pd.DataFrame({"x":range(20),"y":range(20)})
    assert "stability" in ValidationStrategyAgent.propose(df,"UNSUPERVISED")["protocol"]
    assert "seed" in ValidationStrategyAgent.propose(df,"REINFORCEMENT_LEARNING")["checks"][0]

def test_unsupervised_and_rl_engines():
    df=pd.DataFrame({"x":range(30),"y":[v%5 for v in range(30)],"z":[v*v for v in range(30)]})
    u=UnsupervisedAnalysisEngine.run(df)
    r=TabularRLEngine.train(episodes=20)
    assert "pca" in u and "cluster_labels" in u and "anomaly_labels" in u
    assert len(r["episode_rewards"])==20

def test_master_routes_cnn_when_image_paths_are_present():
    df=pd.DataFrame({"image_path":["/tmp/a/cat/1.jpg","/tmp/a/dog/2.jpg"],"label":["cat","dog"]})
    d=MasterDecisionEngine.decide(df,"perform CNN image classification with transfer learning",None,target="label")
    assert d["primary_route"]=="CNN_IMAGE_CLASSIFICATION"
    assert "CNN Image Analysis Agent" in d["sub_agents"]


def test_cnn_folder_scan_requires_multiple_classes(tmp_path):
    from services.cnn_image_analysis import CNNImageAnalysisEngine
    (tmp_path/"cats").mkdir(); (tmp_path/"dogs").mkdir()
    for name in ("a.jpg","b.jpg"):
        (tmp_path/"cats"/name).write_bytes(b"not-a-real-image")
    (tmp_path/"dogs"/"c.jpg").write_bytes(b"not-a-real-image")
    info=CNNImageAnalysisEngine.scan_image_folder(tmp_path)
    assert info["task"]=="image_classification"
    assert set(info["classes"])=={"cats","dogs"}
