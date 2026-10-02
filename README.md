# Data Science Studio Pro V20.5

## Professional AI-assisted data analysis platform

Data Science Studio Pro is designed around an evidence-first workflow for senior Data Scientists. The central intelligence layer is **Agent Data Scientist**, the Master Agent.

The application separates:

- LLM reasoning
- LangGraph orchestration
- analytical memory
- deterministic typed tools
- specialist Agents
- human approval
- evidence/provenance
- visualization
- report generation
- presentation generation

The main GUI/layout is preserved.

## Master Agent

Agent Data Scientist follows:

`Question → Task → Methods → Validation → Approval → Specialist → Evidence → Evaluation → Verification → Stop/Continue`

It considers:

- regression
- classification
- clustering
- unsupervised learning
- anomaly detection
- time series
- statistics
- reinforcement learning
- deep learning
- CNN image analysis

The Master Agent does not use dataset size as a standalone ML/DL decision rule.

## LangGraph + Memory + LLM

### LangGraph
LangGraph is the workflow engine that carries analytical state between stages and human approval gates.

### Memory

**Working memory**: current task, state, evidence and approval status.

**Session/project memory**: datasets, experiments, models, metrics, Sheets, Stories, reports and artifacts.

**Specialist memory**: bounded context for each analytical specialist.

### LLM
The LLM provides bounded task/method review and narrative generation. Deterministic computation is delegated to typed analytical tools and specialist Agents.

## Independent AI-provider selection

The application supports independent LLM/API configuration for:

- Agent Data Scientist
- AI Agent Plot
- AI Agent Report
- AI Agent Presentation

Each Agent checks its own provider immediately before execution through `AgentProviderRegistry`.

This prevents a local model selected for one Agent from silently becoming the provider for another Agent.

## CNN Image Analysis

A new **Data Analysis → CNN Image Analysis** workspace provides a professional computer-vision workflow.

### Workspace capabilities

- ImageFolder dataset discovery
- ResNet-18
- ResNet-50
- VGG-16
- EfficientNet-B0
- pretrained transfer learning
- frozen backbone
- last-block fine-tuning
- full fine-tuning
- image resizing
- augmentation
- class-balanced sampling
- Adam / AdamW
- learning-rate scheduling
- early stopping
- train/validation/test separation
- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- reproducibility metadata
- JSON evidence export
- explainability planning

The CNN specialist is also integrated into Agent Data Scientist through LangGraph, memory and the typed tool registry.

## Sheets

A Sheet is a reproducible analytical view, not simply an image.

A professional Sheet should retain:

- dataset and version
- Rows / Columns / Marks
- filters
- transformations
- chart configuration
- analytical evidence IDs
- annotations
- provenance

## Story

Story is the human-curated presentation layer between analysis and final communication.

Recommended flow:

`Dataset → Analysis → Result → Sheet → Story Point → Report / Presentation`

Story Points can contain selected Sheets, titles, analytical explanations, evidence references and speaker notes. Sheet cards remain movable/resizable.

## Professional Report

The AI Agent Report follows:

`Evidence → Audit → Report Planner → Narrative Engine → Output QA → PDF`

The report should contain coherent sections rather than concatenated metric sentences:

1. Executive Summary
2. Analytical Objective
3. Data & Governance
4. Methodology
5. Validation
6. Results
7. Robustness
8. Visual Findings
9. Discussion
10. Limitations
11. Conclusion
12. Recommendations
13. Reproducibility

## Professional Presentation

The Presentation Agent follows:

`Evidence + Story → Slide Planner → One Message per Slide → Visual Layout → Speaker Notes → PPTX QA`

A slide should answer one analytical question, give visual evidence priority, use concise visible text and retain detailed interpretation in speaker notes.

## Evidence chain

The intended provenance chain is:

`Dataset → Transformation → Method → Validation → Result → Figure → Sheet → Story → Report/Presentation`

This makes analytical communication traceable to the underlying evidence.

## Startup compatibility fix

V20.5 explicitly imports:

```python
from services.agent_provider_registry import AgentProviderRegistry, DEFAULT_AGENTS
```

This fixes the startup error:

`NameError: name 'AgentProviderRegistry' is not defined`

The Story graphics item also uses Qt6-compatible `event.pos()` rather than `QGraphicsSceneMouseEvent.position()`.

## Validation

The V20.5 source was compiled with `python -m compileall` and the full automated test suite passes.
