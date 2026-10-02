"""LangGraph specialist wrapper for unsupervised analysis."""
from services.unsupervised_analysis import UnsupervisedAnalysisEngine

def run_unsupervised_step(df, features=None, seed=42):
    """Execute the bounded unsupervised-analysis engine and return evidence."""
    return UnsupervisedAnalysisEngine.run(df, features=features, seed=seed)
