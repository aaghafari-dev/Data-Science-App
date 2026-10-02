"""Module duty: Extension api.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from .governance import AnalysisExtension, ControlledAnalysisExtensionAPI

__all__ = ["AnalysisExtension", "ControlledAnalysisExtensionAPI"]
