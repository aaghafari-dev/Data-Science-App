# Data Science Studio Pro V20.5.7 — Professional Architecture

## 1. Architectural principle

The application is organized around an evidence-first analytical workflow. The **Agent Data Scientist is the Master Agent**. It does not replace specialist computation; it identifies the analytical task, compares candidate methods, chooses a validation strategy, plans experiments, delegates bounded work, evaluates evidence, and requests human approval at governed transitions.

Main chain:

`Question → Task → Candidate Methods → Validation → Approval → Specialist → Evidence → Evaluation → Verification → Sheets/Story/Report/Presentation`

## 2. Master Agent / LangGraph / Memory / LLM

### Master Agent
`agent/graph_ds.py` is the LangGraph implementation of Agent Data Scientist.

It maintains:
- analytical objective
- target and feature decisions
- task identification
- method candidates
- validation strategy
- specialist queue
- approval state
- evidence IDs
- verification state
- working and specialist memory

### LangGraph
LangGraph controls state transitions. It is the orchestration layer, not the analytical calculator.

Typical governed path:

`PLAN → VALIDATION → PREPROCESSING → MODEL_SELECTION → MODEL_EXECUTION → EVALUATION → ROBUSTNESS → DIAGNOSIS → VERIFY → STOP`

### Memory layers

1. **Working memory** — current LangGraph state and intermediate evidence.
2. **Session/project memory** — datasets, experiments, models, metrics, Sheets, Stories, reports and artifacts.
3. **Specialist memory** — bounded role-specific context for ML, DL, clustering, unsupervised learning, RL and CNN specialists.

Memory stores analytical context and provenance rather than uncontrolled conversation history.

### LLM
The LLM provides bounded reasoning, task/method review and narrative generation. It receives structured evidence and is not allowed to invent targets, data facts, specialist names or analytical tools outside the supplied registry.

## 3. Per-Agent LLM provider architecture

`services/agent_provider_registry.py` provides independent provider selection for:

- Agent Data Scientist
- AI Agent Plot
- AI Agent Report
- AI Agent Presentation

Every governed Agent performs an execution-time provider preflight immediately before starting. A local checkpoint and an API model are therefore not accidentally shared between Agents.

## 4. Specialist Agents

Current specialist families include:

- Data Quality Agent
- Statistical Insight Agent
- ML Agent
- DL Agent
- Clustering Agent
- Unsupervised Learning Agent
- Reinforcement Learning Agent
- Anomaly Detection Agent
- Time-Series Agent
- CNN Image Analysis Agent

The Master Agent selects specialists through the method registry and approval workflow.

## 5. CNN Image Analysis

### Workspace
`CNN Image Analysis` is a new submenu under **Data Analysis**. It opens a professional dialog and does not change the main GUI geometry.

Workspace sections cover:

1. Dataset & Training
2. Dataset & Split
3. Training & Fine-tuning
4. Evaluation & Diagnostics
5. Explainability
6. Reproducibility & Export

Supported controls include:
- ImageFolder dataset discovery
- ResNet-18
- ResNet-50
- VGG-16
- EfficientNet-B0
- pretrained transfer learning
- freeze-backbone training
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
- checkpoint/evidence export
- reproducibility metadata
- explainability planning

### Master Agent integration
CNN is a first-class Master-Agent route:

`CNN_IMAGE_CLASSIFICATION → CNN Image Analysis Agent → Evidence DAG`

The route is triggered by explicit image/CNN objectives or detected image-path evidence. It is not selected by a simple dataset-size threshold.

## 6. Sheets

A Sheet is a reproducible analytical view, not merely a screenshot. Its provenance should retain:

- dataset/version
- Rows / Columns / Marks
- filters
- chart configuration
- transformations
- analytical evidence IDs
- annotations
- source specialist/analysis

The existing GUI/layout is preserved.

## 7. Story

Story is the human-curated narrative layer between analytical Sheets and final presentation.

Recommended chain:

`Dataset → Analysis → Result → Sheet → Story Point → Report / Presentation`

A Story Point should contain a title, selected Sheets, explanatory text, evidence references and optional speaker notes. Drag/drop ordering and movable/resizable Sheet cards are preserved.

## 8. Report pipeline

`Evidence DAG → Evidence Audit → Report Planner → Narrative Engine → Output QA → PDF`

The narrative engine generates connected section-level prose rather than concatenating independent metric sentences.

Recommended sections:
- Executive Summary
- Analytical Objective
- Data & Governance
- Methodology
- Validation
- Results
- Robustness
- Visual Findings
- Discussion
- Limitations
- Conclusion
- Recommendations
- Reproducibility

## 9. Presentation pipeline

`Evidence + Story → Presentation Planner → One Analytical Message per Slide → Visual Layout → Speaker Notes → PPTX QA`

The Presentation Agent should not simply copy PDF paragraphs into slides. Each slide should answer one analytical question, prioritize the visualization, keep visible text concise, and place detailed interpretation in speaker notes.

## 10. Validation and challenger strategy

The Master Agent explicitly separates:

`What is the task? → How should it be validated? → Which method should be compared?`

For supervised problems, the architecture supports baseline + primary + challenger thinking. Validation can be IID, stratified, grouped, temporal, unsupervised stability, or RL policy evaluation depending on the task.

## 11. Quality and safety

Before final output, the system should verify:
- reported metrics exist in evidence
- figures have evidence references
- conclusions do not exceed evidence
- causal language is not introduced without causal design
- validation assumptions are visible
- provenance and approval records are retained
- computational resources are checked before expensive local-model execution

## 12. Important compatibility fixes

The reported V20.5.5 startup failure is fixed: `_load_dropped_datasets()` is now defined and is connected to the Loaded Sheets drop handler. The handler delegates to the canonical dataset ingestion path.

The previous `AgentProviderRegistry` import and Qt6 graphics-scene compatibility fixes are retained.


## V20.5.7 Resource Profile and Output Governance

V20.5.7 adds a professional per-Agent local-LLM resource profile. The **Select Local LLM model or API key** dialog allows an independent RAM budget and GPU-VRAM budget for each governed Agent, with an Auto-detect option. The selected budget is an execution constraint, not a replacement for physical hardware detection. Before Transformers loads a local checkpoint, the application estimates CPU and GPU working-set requirements from measured checkpoint files and dtype/quantization metadata when available, then compares them with both the configured budget and currently free system/GPU memory.

This prevents the common misconception that a model name such as Qwen2.5-7B implies a 2-GB GPU requirement. A 7B model stored in FP16 has roughly 14 GB of raw weights before runtime overhead; a 4-bit checkpoint can be much smaller. Actual feasibility therefore depends on checkpoint precision, runtime buffers, context/cache and available memory.

The Presentation Agent uses stable scalar slide titles to index rendered figures; dictionaries are never used as dictionary keys. This fixes the `Unhashable type: 'dict'` failure.

### Professional communication pipeline

`Evidence DAG → Evidence Audit → Report/Plot Planner → coherent narrative → QA → Story → Presentation`

Sheets remain reproducible analytical objects; Story is the human-controlled narrative layer; Report is the evidence-bound scientific document; Presentation is the audience-oriented communication layer.


## V20.5.7 — Professional Local-LLM Resource Profiles

The **Select Local LLM model or API key** workspace now provides an independent resource profile for each governed Agent:

- RAM: Auto-detect, 8, 16, 24, 32, 48, 64, 128 GiB
- GPU VRAM: Auto-detect, 2, 4, 8, 12, 16, 18, 24, 32, 48, 64 GiB

The selected values are an explicit resource budget. They do not override the physical machine. Immediately before local-model execution, the application also measures current free system RAM and CUDA VRAM where available. The model checkpoint is inspected for actual weight-file size and dtype/quantization metadata when possible. The application then estimates a CPU and GPU inference working set with runtime headroom and blocks unsafe execution before Transformers performs a large allocation.

### Why 2 GB is not a universal requirement for Qwen2.5-7B

A model's parameter count, quantization format, runtime and context length matter. Qwen2.5-7B has about 7.61B parameters. A normal FP16/BF16 representation is far larger than 2 GiB before runtime overhead. Quantized 4-bit Qwen2.5-7B checkpoints are much smaller, but the official Hugging Face repositories are still multi-gigabyte model files and the actual GPU requirement depends on the inference stack and available cache. Therefore the application deliberately does **not** assume that a generic statement such as “2 GB is enough” is safe for every Qwen2.5-7B configuration.

## V20.5.7 — Presentation Agent Reliability

The Presentation Agent no longer converts `(slide_dict, path)` pairs into a dictionary. A slide dictionary is not hashable, which caused the previous `Unhashable type: 'dict'` failure. Rendered figures are now indexed by a stable scalar slide title, and a regression test protects this behaviour.

## V20.5.7 — Story as a Professional Communication Workspace

Story now includes:

- rich-text editing;
- font family;
- font size;
- bold / italic / underline;
- text color;
- movable/reorderable Story points;
- presentation mode with Previous / Next navigation;
- full-screen presentation view;
- Sheet-linked visual evidence;
- PNG/PDF export.

The intended communication chain is:

`Analysis → Sheet → Story Point → Story → Report / Presentation`

The user remains the author of the scientific message. AI assists with narrative generation and presentation design, but the approved Story remains the primary human-curated communication source.

## V20.5.7 — Sheet Synchronization

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

The project includes `Qt_for_Python_PySide6_10min_Professional_Learning.pptx`, a 10-minute teaching presentation covering QApplication, widgets, layouts, signals/slots, Model/View, dialogs, Graphics View, worker threads, architecture, quality checks and a short student challenge. The teaching material follows the official Qt for Python / PySide6 documentation and Qt 6 tutorials.


## V20.5.7 — MLP convergence and Windows runtime hardening

The tabular scikit-learn MLP implementation is centralized in `services/mlp_training.py`. Supervised Analysis, the ML Agent and the DL Agent use the same explicit policy: `(64, 32)` hidden layers, `max_iter=1500`, early stopping, `n_iter_no_change=30`, `tol=1e-4` and `validation_fraction=0.15`.

Convergence warnings are captured as structured model evidence rather than globally suppressed. Direct DL CPU/scikit-learn training retries once at `max_iter=3000` when the first fit reaches the iteration limit. Cross-validation searches are not blindly retried; their convergence-warning count and samples are recorded for auditability.

On Windows, `LOKY_MAX_CPU_COUNT` is explicitly bounded to the logical CPU count to avoid the joblib/loky physical-core probe `WinError 2` on systems where that Windows utility is unavailable. This does not change the application's bounded parallelism policy.


## V20.5.7 Professional Upgrade

### Agent approval and transparency
Every consequential Agent Data Scientist transition requires explicit human approval. Each gate presents: (1) what was done, (2) evidence/results available so far, (3) what the Agent proposes next, (4) risks/assumptions, and (5) Approve / Reject / Abort. Rejection returns the workflow to a revision state rather than silently continuing.

### Agent results
The latest Master-Agent results are published through **AI Agents → Agent Results & Evidence**. Specialist outputs, approvals, evidence IDs, diagnostics and governance records are retained there. Classification and Regression workspaces expose Methods & Model Selection, Evaluation & Diagnostics, Deployment & Monitoring, and exportable result evidence.

### AI Agent Plot and AI Agent Report
Every AI-generated visualization now carries a title, caption, explanation, analytical rationale and evidence-linked findings. Reports reuse those plots and include the same title/caption/explanation in the Visual Results section.

### Story and Presentation
Story remains the Tableau-like human-curated narrative layer. Presentation Mode is now an independent, full-screen slide viewer rather than a child overlay of the Story editor. The viewer has a slide canvas, story-point title, page counter, navigation and exit control. AI Agent Presentation creates a real PPTX and verifies that the output file exists before reporting success.

### Local LLM resource reuse
If the same local model is already loaded by Agent Data Scientist, AI Agent Plot/Report/Presentation reuse the resident model instead of incorrectly preflighting its own memory a second time. This prevents false RAM-insufficient errors caused by counting the application's already-loaded Qwen checkpoint against itself.

### Transformers warnings
Local generation uses a deterministic generation configuration, avoids simultaneously passing stale `max_length` and `max_new_tokens`, and disables destructive BPE cleanup. This removes the generation warnings observed with Qwen2Tokenizer.

### Loaded Sheets drag-and-drop
Supported local datasets can be dropped directly onto Loaded Sheets. The drop path goes through the same dataset card, lineage, contract and fingerprint registration as File → Open.

### Extra Insight sorting
Extra Insight provides Ascending/Descending sorting in its control bar. Numeric and categorical columns use the same explicit ordering policy.

### Correlation Matrix
Correlation Matrix is presented as a professional heatmap with numeric annotations and a color scale rather than as a raw text matrix. The interface explicitly reminds the user that correlation is not causality.

### Plot legends
Plot rendering reserves layout space whenever an external legend exists, preventing legends from being clipped or disappearing after the final layout pass.

### Governance
Governance submenus provide reproducibility and risk controls: Leakage Gate prevents future/derived information from silently entering models; Dataset Card records dataset meaning/provenance; Model Cards document model behaviour; Experiment Comparison supports challenger review; Model Promotion controls lifecycle transitions; Data Diff and Schema Drift detect changes; Evidence DAG links conclusions to evidence; Agent Evaluation audits Agent behaviour; Split Wizard makes validation design explicit; Error Analysis examines failures; Statistical Analysis provides inferential workflows; Monitoring detects post-deployment drift; Publication Package creates reproducible outputs; Tool Registry governs Agent capabilities; Analysis Recipe supports replay; State Machine records lifecycle; Artifact Registry/Lineage tracks generated objects. These are required because a high-level data-science application must make not only results but also assumptions, provenance, approvals and changes reviewable.

### Sharing
Sharing should preserve evidence and provenance. Export PNG is a visual snapshot; Email Current Sheet shares a view; Email Story shares the curated narrative; Streamlit export creates an interactive shareable view; Collaboration Package bundles datasets/recipes/evidence/artifacts according to privacy and governance rules. Share actions should never silently expose API keys or private credentials.


## 10. V20.5.7 Human-Gated Execution and Memory

### Approval contract
`PLAN → approval → VALIDATION → approval → PREPROCESSING → approval → MODEL_SELECTION → approval → MODEL_EXECUTION → approval → EVALUATION → approval → ROBUSTNESS → approval → DIAGNOSIS → approval → VERIFY → approval → STOP`

Every approval package contains `completed_bullets`, `approval_result_points`, `next_bullets` and `risk_bullets`. The GUI presents these as separate bullet-point sections without changing the main application layout.

### Memory layers
- **LangGraph state:** current run state and deterministic transition data.
- **AgentMemory:** bounded role-specific working memory.
- **AgentMemoryStore:** secret-free persistent approval/run audit records.
- **Evidence DAG:** analytical provenance and evidence relationships.

A stable `run_id` identifies one analysis. A stable `thread_id` identifies the corresponding LangGraph conversation context. API keys are never written to persistent Agent memory.

### Dataset ingestion
`MainWindow._load_dataset_path()` is the canonical ingestion path. `open_file()` and `_load_dropped_datasets()` both call it. This guarantees identical governance for File → Open and Loaded Sheets drag-and-drop.

## 11. Professional application recommendations

### Agents
Use a common typed Agent contract: `prepare → self_check → approval → execute → evidence → evaluate → approval`. Each specialist should expose assumptions, inputs, outputs, warnings, runtime and evidence references.

### LangGraph connections
Keep orchestration state separate from long-term memory. Use stable run/thread IDs, explicit node names, typed state and evidence references. A future production extension can attach a durable LangGraph checkpointer without changing the GUI.

### Story
Treat Story Points as human-curated communication objects linked to reproducible Sheets and evidence IDs, not screenshots.

### Toolbar
Toolbar actions should always act on the active dataset/sheet and should never silently change analysis state. Destructive actions should be undoable or explicitly confirmed.

### Data Analysis
Each submenu should expose: objective, assumptions, inputs, method, result, diagnostics, limitations and export/evidence actions. Correlation should be rendered as a heatmap with numeric annotations rather than a raw table.

### Sharing
Sharing should distinguish image-only export, analytical view export, Story export, interactive application export and full collaboration package. Secrets must never be included.


## 20. V20.5.7 Story and Runtime Architecture

### Story object model
The Story workspace stores an editable communication object per Story point:

`Sheet → Story Point → Figure Title + Rich Caption + Formatting + Position/Size → Story → Presentation/Export`

Story state includes order, title, captions, formatting, background, card geometry and Sheet linkage. Rebuilding a Story preview must never silently discard user-edited geometry or narrative content.

### Presentation lifecycle safety
The independent Story presentation viewer owns its rendering callbacks. A close/destroy event marks the viewer inactive before delayed callbacks can update Qt widgets. Rendering checks lifecycle state before accessing QLabel/QWidget objects.

### Shared local LLM runtime
`services/llm_config.py` owns the shared local loader used by all governed Agents. `agent/local_llm.py` distinguishes initial loading from resident reuse.

Provider selection remains per-Agent through `AgentProviderRegistry`; model residency is intentionally shared at runtime. This gives independent governance without duplicating a large model in memory.

### Generation safety
`GenerationConfig` is normalized before pipeline creation. Sampling-only flags are removed for deterministic generation, `max_length` is not combined with `max_new_tokens`, and Qwen BPE token cleanup is disabled.

### Recommended UX architecture
The professional UX roadmap prioritizes reusable Analysis Templates, Question-to-Evidence search, reproducibility replay, uncertainty-first visualization, claim review, autosave/recovery, accessibility, Agent benchmarking, compute-budget visibility and self-describing collaboration packages.
