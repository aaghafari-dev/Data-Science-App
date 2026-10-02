"""LangGraph specialist wrapper for reinforcement learning."""
from services.reinforcement_learning import TabularRLEngine

def run_reinforcement_step(**kwargs):
    """Execute the bounded tabular RL specialist."""
    return TabularRLEngine.train(**kwargs)
