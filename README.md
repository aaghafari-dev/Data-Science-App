# Data Science App 

## Professional AI-assisted data analysis platform

Data Science Studio Pro is designed around an evidence-first workflow for senior Data Scientists. The central intelligence layer is **Agent Data Scientist**, the Master Agent.

The application separates:

* LLM reasoning
* LangGraph orchestration
* analytical memory
* deterministic typed tools
* specialist Agents
* human approval
* evidence/provenance
* visualization
* report generation
* presentation generation

The main GUI/layout is preserved.

## Master Agent

Agent Data Scientist follows:

`Question → Task → Methods → Validation → Approval → Specialist → Evidence → Evaluation → Verification → Stop/Continue`

It considers:

* regression
* classification
* clustering
* unsupervised learning
* anomaly detection
* time series
* statistics
* reinforcement learning
* deep learning
* CNN image analysis

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

* Agent Data Scientist
* AI Agent Plot
* AI Agent Report
* AI Agent Presentation

Each Agent checks its own provider immediately before execution through `AgentProviderRegistry`.

This prevents a local model selected for one Agent from silently becoming the provider for another Agent.

## CNN Image Analysis

A new **Data Analysis → CNN Image Analysis** workspace provides a professional computer-vision workflow.

### Workspace capabilities

* ImageFolder dataset discovery
* ResNet-18
* ResNet-50
* VGG-16
* EfficientNet-B0
* pretrained transfer learning
* frozen backbone
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
* reproducibility metadata
* JSON evidence export
* explainability planning

The CNN specialist is also integrated into Agent Data Scientist through LangGraph, memory and the typed tool registry.

## Sheets

A Sheet is a reproducible analytical view, not simply an image.

A professional Sheet should retain:

* dataset and version
* Rows / Columns / Marks
* filters
* transformations
* chart configuration
* analytical evidence IDs
* annotations
* provenance

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
3. Data \& Governance
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

## Validation

The source was compiled with `python -m compileall` and the full automated test suite passes.



## Resource Profile and Output Governance

The **Select Local LLM model or API key** dialog allows an independent RAM budget and GPU-VRAM budget for each governed Agent, with an Auto-detect option. The selected budget is an execution constraint, not a replacement for physical hardware detection. Before Transformers loads a local checkpoint, the application estimates CPU and GPU working-set requirements from measured checkpoint files and dtype/quantization metadata when available, then compares them with both the configured budget and currently free system/GPU memory.

This prevents the common misconception that a model name such as Qwen2.5-7B implies a 2-GB GPU requirement. A 7B model stored in FP16 has roughly 14 GB of raw weights before runtime overhead; a 4-bit checkpoint can be much smaller. Actual feasibility therefore depends on checkpoint precision, runtime buffers, context/cache and available memory.

The Presentation Agent uses stable scalar slide titles to index rendered figures; dictionaries are never used as dictionary keys. This fixes the `Unhashable type: 'dict'` failure.

### Professional communication pipeline

`Evidence DAG → Evidence Audit → Report/Plot Planner → coherent narrative → QA → Story → Presentation`

Sheets remain reproducible analytical objects; Story is the human-controlled narrative layer; Report is the evidence-bound scientific document; Presentation is the audience-oriented communication layer.



## Professional Local-LLM Resource Profiles

The **Select Local LLM model or API key** workspace now provides an independent resource profile for each governed Agent:

* RAM: Auto-detect, 8, 16, 24, 32, 48, 64, 128 GiB
* GPU VRAM: Auto-detect, 2, 4, 8, 12, 16, 18, 24, 32, 48, 64 GiB

The selected values are an explicit resource budget. They do not override the physical machine. Immediately before local-model execution, the application also measures current free system RAM and CUDA VRAM where available. The model checkpoint is inspected for actual weight-file size and dtype/quantization metadata when possible. The application then estimates a CPU and GPU inference working set with runtime headroom and blocks unsafe execution before Transformers performs a large allocation.

### Why 2 GB is not a universal requirement for Qwen2.5-7B

A model's parameter count, quantization format, runtime and context length matter. Qwen2.5-7B has about 7.61B parameters. A normal FP16/BF16 representation is far larger than 2 GiB before runtime overhead. Quantized 4-bit Qwen2.5-7B checkpoints are much smaller, but the official Hugging Face repositories are still multi-gigabyte model files and the actual GPU requirement depends on the inference stack and available cache. Therefore the application deliberately does **not** assume that a generic statement such as “2 GB is enough” is safe for every Qwen2.5-7B configuration.

## Presentation Agent Reliability

The Presentation Agent no longer converts `(slide\_dict, path)` pairs into a dictionary. A slide dictionary is not hashable, which caused the previous `Unhashable type: 'dict'` failure. Rendered figures are now indexed by a stable scalar slide title, and a regression test protects this behaviour.

## Story as a Professional Communication Workspace

Story now includes:

* rich-text editing;
* font family;
* font size;
* bold / italic / underline;
* text color;
* movable/reorderable Story points;
* presentation mode with Previous / Next navigation;
* full-screen presentation view;
* Sheet-linked visual evidence;
* PNG/PDF export.

The intended communication chain is:

`Analysis → Sheet → Story Point → Story → Report / Presentation`

The user remains the author of the scientific message. AI assists with narrative generation and presentation design, but the approved Story remains the primary human-curated communication source.

## Sheet Synchronization

Every plot refresh persists the active Sheet's current Rows, Columns, Marks, chart type, aggregations and dataset snapshot. A newly created Sheet is intentionally blank and is activated immediately so the user can begin a new analysis. Story rendering reads the latest Sheet snapshot instead of a stale copy.

## Professional Recommendations

### Agent Data Scientist

The Master Agent should continue to operate as a senior analytical supervisor:

`Objective → Task → Leakage Gate → Validation Strategy → Method Comparison → Baseline/Primary/Challenger → Approval → Specialist → Evidence → Robustness → Diagnosis → Independent Verification`

Priority improvements are stronger validation-strategy selection, group/time-aware validation, explicit baseline/challenger experiments, uncertainty calibration and a dedicated analytical QA stage.

### AI Agent Plot

The Plot Agent should answer an analytical question rather than generate a gallery. Each figure should have a purpose, evidence source, fields, quantity, interpretation and limitations. Visual associations must remain explicitly non-causal unless inferential evidence supports a stronger statement.

### AI Agent Report

The Report Agent uses evidence audit → report planning → section-level narrative → QA → PDF. Executive narrative is separated from raw machine-readable evidence so the report remains readable while retaining auditability.

### AI Agent Presentation

The Presentation Agent uses the approved Story plus Evidence DAG. Each slide should communicate one analytical message, use the visualization as the visual focus, keep visible text concise, and put technical detail into speaker notes. A final QA stage checks title, purpose, text density and visual context.

## Qt for Python Learning Material

The project includes `Qt\_for\_Python\_PySide6\_10min\_Professional\_Learning.pptx`, a 10-minute teaching presentation covering QApplication, widgets, layouts, signals/slots, Model/View, dialogs, Graphics View, worker threads, architecture, quality checks and a short student challenge. The teaching material follows the official Qt for Python / PySide6 documentation and Qt 6 tutorials.



## MLP convergence and Windows runtime hardening

The tabular scikit-learn MLP implementation is centralized in `services/mlp\_training.py`. Supervised Analysis, the ML Agent and the DL Agent use the same explicit policy: `(64, 32)` hidden layers, `max\_iter=1500`, early stopping, `n\_iter\_no\_change=30`, `tol=1e-4` and `validation\_fraction=0.15`.

Convergence warnings are captured as structured model evidence rather than globally suppressed. Direct DL CPU/scikit-learn training retries once at `max\_iter=3000` when the first fit reaches the iteration limit. Cross-validation searches are not blindly retried; their convergence-warning count and samples are recorded for auditability.

On Windows, `LOKY\_MAX\_CPU\_COUNT` is explicitly bounded to the logical CPU count to avoid the joblib/loky physical-core probe `WinError 2` on systems where that Windows utility is unavailable. This does not change the application's bounded parallelism policy.



## 

