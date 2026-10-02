# Data Science Studio Pro V20.5 — Professional Architecture

## 1\. Architectural principle

The application is organized around an evidence-first analytical workflow. The **Agent Data Scientist is the Master Agent**. It does not replace specialist computation; it identifies the analytical task, compares candidate methods, chooses a validation strategy, plans experiments, delegates bounded work, evaluates evidence, and requests human approval at governed transitions.

Main chain:

`Question → Task → Candidate Methods → Validation → Approval → Specialist → Evidence → Evaluation → Verification → Sheets/Story/Report/Presentation`

## 2\. Master Agent / LangGraph / Memory / LLM

### Master Agent

`agent/graph\_ds.py` is the LangGraph implementation of Agent Data Scientist.

It maintains:

* analytical objective
* target and feature decisions
* task identification
* method candidates
* validation strategy
* specialist queue
* approval state
* evidence IDs
* verification state
* working and specialist memory

### LangGraph

LangGraph controls state transitions. It is the orchestration layer, not the analytical calculator.

Typical governed path:

`PLAN → VALIDATION → PREPROCESSING → MODEL\_SELECTION → MODEL\_EXECUTION → EVALUATION → ROBUSTNESS → DIAGNOSIS → VERIFY → STOP`

### Memory layers

1. **Working memory** — current LangGraph state and intermediate evidence.
2. **Session/project memory** — datasets, experiments, models, metrics, Sheets, Stories, reports and artifacts.
3. **Specialist memory** — bounded role-specific context for ML, DL, clustering, unsupervised learning, RL and CNN specialists.

Memory stores analytical context and provenance rather than uncontrolled conversation history.

### LLM

The LLM provides bounded reasoning, task/method review and narrative generation. It receives structured evidence and is not allowed to invent targets, data facts, specialist names or analytical tools outside the supplied registry.

## 3\. Per-Agent LLM provider architecture

`services/agent\_provider\_registry.py` provides independent provider selection for:

* Agent Data Scientist
* AI Agent Plot
* AI Agent Report
* AI Agent Presentation

Every governed Agent performs an execution-time provider preflight immediately before starting. A local checkpoint and an API model are therefore not accidentally shared between Agents.

## 4\. Specialist Agents

Current specialist families include:

* Data Quality Agent
* Statistical Insight Agent
* ML Agent
* DL Agent
* Clustering Agent
* Unsupervised Learning Agent
* Reinforcement Learning Agent
* Anomaly Detection Agent
* Time-Series Agent
* CNN Image Analysis Agent

The Master Agent selects specialists through the method registry and approval workflow.

## 5\. CNN Image Analysis

### Workspace

`CNN Image Analysis` is a new submenu under **Data Analysis**. It opens a professional dialog and does not change the main GUI geometry.

Workspace sections cover:

1. Dataset \& Training
2. Dataset \& Split
3. Training \& Fine-tuning
4. Evaluation \& Diagnostics
5. Explainability
6. Reproducibility \& Export

Supported controls include:

* ImageFolder dataset discovery
* ResNet-18
* ResNet-50
* VGG-16
* EfficientNet-B0
* pretrained transfer learning
* freeze-backbone training
* last-block fine-tuning
* full fine-tuning
* image resizing
* augmentation
* class-balanced sampling
* Adam / AdamW
* learning-rate scheduling
* early stopping
* train/validation/test separation
* accuracy
* balanced accuracy
* macro precision
* macro recall
* macro F1
* confusion matrix
* checkpoint/evidence export
* reproducibility metadata
* explainability planning

### Master Agent integration

CNN is a first-class Master-Agent route:

`CNN\_IMAGE\_CLASSIFICATION → CNN Image Analysis Agent → Evidence DAG`

The route is triggered by explicit image/CNN objectives or detected image-path evidence. It is not selected by a simple dataset-size threshold.

## 6\. Sheets

A Sheet is a reproducible analytical view, not merely a screenshot. Its provenance should retain:

* dataset/version
* Rows / Columns / Marks
* filters
* chart configuration
* transformations
* analytical evidence IDs
* annotations
* source specialist/analysis

The existing GUI/layout is preserved.

## 7\. Story

Story is the human-curated narrative layer between analytical Sheets and final presentation.

Recommended chain:

`Dataset → Analysis → Result → Sheet → Story Point → Report / Presentation`

A Story Point should contain a title, selected Sheets, explanatory text, evidence references and optional speaker notes. Drag/drop ordering and movable/resizable Sheet cards are preserved.

## 8\. Report pipeline

`Evidence DAG → Evidence Audit → Report Planner → Narrative Engine → Output QA → PDF`

The narrative engine generates connected section-level prose rather than concatenating independent metric sentences.

Recommended sections:

* Executive Summary
* Analytical Objective
* Data \& Governance
* Methodology
* Validation
* Results
* Robustness
* Visual Findings
* Discussion
* Limitations
* Conclusion
* Recommendations
* Reproducibility

## 9\. Presentation pipeline

`Evidence + Story → Presentation Planner → One Analytical Message per Slide → Visual Layout → Speaker Notes → PPTX QA`

The Presentation Agent should not simply copy PDF paragraphs into slides. Each slide should answer one analytical question, prioritize the visualization, keep visible text concise, and place detailed interpretation in speaker notes.

## 10\. Validation and challenger strategy

The Master Agent explicitly separates:

`What is the task? → How should it be validated? → Which method should be compared?`

For supervised problems, the architecture supports baseline + primary + challenger thinking. Validation can be IID, stratified, grouped, temporal, unsupervised stability, or RL policy evaluation depending on the task.

## 11\. Quality and safety

Before final output, the system should verify:

* reported metrics exist in evidence
* figures have evidence references
* conclusions do not exceed evidence
* causal language is not introduced without causal design
* validation assumptions are visible
* provenance and approval records are retained
* computational resources are checked before expensive local-model execution

## 

