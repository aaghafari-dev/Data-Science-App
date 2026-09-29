# Data Science Studio Pro — Project Structure V18.6

## Engineering objective

A professional Windows/PyQt6 data-science desktop environment for high-level analytical users. The application combines a Tableau-like analytical surface, governed ML/DL agents, LangGraph orchestration, professional model evaluation, leakage diagnostics, reproducible artifacts, statistical analysis, data-quality controls, and evidence-bound reporting.

**GUI rule:** the existing GUI geometry, menus, shelves, Marks panel, dialogs and layout are preserved. V18.6 adds backend intelligence and improves existing behavior without redesigning the main interface.

## V18.6 focus

1. **Professional ML/DL evaluation with uncertainty and robustness testing** rather than a single holdout metric.
2. **Hardware-aware compute selection** with independent NVIDIA hardware, PyTorch CUDA, architecture and runtime validation.
3. **Plan → Critic → Execute** control inside the Master Agent.
4. **Feature Stability Analysis** using repeated resampling and permutation importance.
5. **Temporal Availability Matrix** for feature availability over time.
6. **Counterfactual Leakage Test** for suspicious features.
7. **Automatic Model Diagnosis** from evaluation and leakage evidence.
8. **Agent Self-Check** before every human approval gate.
9. **Agent Why** as a first-class evidence object; only decision-level rationale is stored, not private chain-of-thought.
10. **Evidence-first integration** so Report, Presentation, Model Cards, Publication and Sharing can consume the same evidence objects.

## Top-level structure

```text
app/
├── main.py
├── config.py
├── requirements.txt
├── requirements-optional.txt
├── README.md
├── PROMPT.txt
├── prompt.txt
├── Project Structure.md
├── Mermaid Diagram.mmd
├── Mermaid Diagram.png
├── Dockerfile
├── dvc.yaml
├── .github/workflows/ci.yml
├── assets/
├── core/
│   ├── data_engine.py
│   ├── viz_engine.py
│   ├── tableau_views.py
│   ├── sheet_manager.py
│   ├── sheet_manager_V3.py
│   └── theme_manager.py
├── agent/
│   ├── graph_ds.py                 # Master Agent / LangGraph plan-critic-execute control
│   ├── graph_ml.py                 # ML Agent + professional evaluation/diagnostics
│   ├── graph_dl.py                 # DL Agent + hardware-aware compute/evaluation
│   ├── graph_plot.py               # LangGraph Plot Planner + Renderer
│   ├── graph_report.py             # Evidence-bound Report Agent
│   ├── graph_presentation.py       # Streamlit Presentation Agent
│   ├── graph_table_creation.py     # Web/API table acquisition Agent
│   ├── graph_question_analysis.py  # Question → Analysis Agent
│   ├── local_llm.py
│   ├── prompts.py
│   ├── tools.py
│   └── voice_layer.py
├── services/
│   ├── agent_quality.py            # Evaluation, stability, temporal, leakage, diagnosis, Why, self-check
│   ├── compute_backend.py          # Hardware + PyTorch CUDA + runtime validation
│   ├── target_feature_selection.py
│   ├── model_preprocessing.py
│   ├── split_wizard.py
│   ├── error_analysis.py
│   ├── statistical_analysis.py
│   ├── duckdb_workspace.py
│   ├── pdf_tables.py
│   ├── macro_library.py
│   ├── agent_tools.py
│   ├── agent_console.py
│   ├── agent_memory.py
│   ├── analysis_recipe.py
│   ├── analysis_state.py
│   ├── artifacts.py
│   ├── governance.py
│   ├── experiment_registry.py
│   ├── model_registry.py
│   ├── monitoring.py
│   ├── lineage.py
│   ├── explainability.py
│   ├── profiling.py
│   ├── privacy.py
│   ├── notebook_export.py
│   ├── publication_package.py
│   ├── sharing.py
│   ├── large_data.py
│   ├── api_service.py
│   └── extension_api.py
└── tests/
    ├── test_agent_quality.py
    ├── test_viz_engine.py
    ├── test_data_engine.py
    ├── test_services.py
    ├── test_governance.py
    ├── test_professional_features.py
    ├── test_v182_features.py
    ├── test_v183_features.py
    ├── test_v184_features.py
    └── test_agent_imports.py
```

## Agent architecture

The application still has exactly three core data-science agents:

```text
                    Master Agent
                         │
              Plan → Critic → Self-Check
                         │
                  Human Approval
                    /          \
                  ML            DL
                  │              │
          professional      professional
           evaluation        evaluation
                  \              /
                   diagnostics
                         │
                  Agent Plot
                         │
                  Agent Report
                         │
               Agent Presentation
```

LangGraph remains the orchestration engine. Agent Benchmarking evaluates behavior around the graph; it does not replace LangGraph.

## Master Agent: Plan → Critic → Execute

Each executable milestone now has an explicit plan containing:

- objective
- inputs
- expected output
- risk
- route/target
- tool budget
- human-gate requirement

The Critic checks:

- target exists and is valid
- leakage gate is not blocked
- predictors are available
- tool budget is bounded
- human approval is required

Only then is the proposal exposed for approval. A rejection creates a revised proposal; it is never silently converted into approval.

## Agent Self-Check

Before every approval, the graph produces an `agent_self_check` object. It records:

- plan completeness
- evidence binding
- bounded action/tool budget
- human-gate status
- blocking reasons

The self-check is stored in the Evidence DAG and Agent Run Console.

## Agent Why

`AgentWhyEvidence` creates a first-class `agent_why` evidence object containing:

- agent
- decision
- concise decision-level rationale
- evidence references
- alternatives considered
- constraints
- disclosure that private chain-of-thought is not stored

This allows a professional user to understand **why the agent selected an action** without exposing or depending on hidden chain-of-thought.

## Professional ML/DL evaluation with uncertainty and robustness testing

### ML

Each candidate model now records:

- held-out metrics
- baseline comparison
- repeated cross-validation
- mean/std/min/max CV behavior
- train/test generalization gap
- bootstrap uncertainty where supported
- classification calibration diagnostics
- existing error analysis

The model is still selected using the declared primary held-out metric, but the user receives the complete evaluation bundle rather than a single number.

### DL

The DL agent records:

- held-out metrics
- baseline comparison
- compute backend
- runtime diagnostics
- temporal availability evidence
- model diagnosis
- safe CUDA/CPU fallback information

The current bounded neural path intentionally remains conservative for a 2 GB-class GPU.

## Advanced diagnostics

### Feature Stability Analysis

Repeated resampling + permutation importance estimates whether feature importance remains positive and consistent across runs. It is diagnostic evidence, not a reason to automatically delete a feature.

### Temporal Availability Matrix

A detected date/time field is divided into chronological quantile buckets. The service records feature non-missing availability for each period and flags features that appear materially later in the timeline.

A late-appearing feature is **not automatically declared leakage**. The user must verify whether it was available at prediction time.

### Counterfactual Leakage Test

For suspicious features, the service trains the same model protocol with the feature removed and compares performance on the same holdout split. A material performance drop is recorded as model dependence and triggers provenance review; it is not treated as causal proof of leakage.

### Automatic Model Diagnosis

The diagnostic layer looks for:

- generalization gaps
- poor calibration
- unstable feature importance
- late temporal availability
- material counterfactual dependence

It returns explicit hypotheses and suggested next checks rather than an opaque quality score.

## Compute backend

`services/compute_backend.py` separates:

1. **Hardware detection** — Windows `nvidia-smi`.
2. **PyTorch CUDA build detection** — `torch.version.cuda`.
3. **CUDA availability** — `torch.cuda.is_available()`.
4. **Architecture compatibility** — `torch.cuda.get_device_capability()` and `get_arch_list()`.
5. **Runtime validation** — isolated CUDA smoke test in a subprocess.
6. **VRAM policy** — bounded execution for low-VRAM devices.

Therefore, a system with an NVIDIA GPU but a CPU-only PyTorch build is reported as:

> NVIDIA hardware detected, but PyTorch CUDA is not currently usable.

rather than incorrectly reporting that the physical GPU does not exist.

For an MX250-class 2 GB device, the application treats the GPU as a small accelerator. Large dense tabular tensors and large models are routed to CPU rather than risking CUDA out-of-memory failures.

The three user-facing policies remain:

- **CPU** — CPU execution only.
- **CPU+GPU** — opportunistic hybrid policy; CPU remains safe when CUDA is unavailable.
- **GPU** — CUDA is required; if CUDA validation fails, the application refuses unsafe GPU execution and falls back safely.

## Existing V18.4 functionality retained

- Tableau-like Rows/Columns and Marks behavior.
- Multiple Color/Text/Detail/Tooltip fields; single Size field.
- Plot formatting context menu and PNG/JPG/PDF export.
- Pie chart.
- Automatic target validation and A/A² direction ambiguity detection.
- AI Agent Plot as a genuine LangGraph planning agent.
- Evidence-bound Report Agent.
- Presentation Agent consuming Data Scientist + Plot + Report evidence.
- Data Contracts and Schema Drift.
- Leakage-aware Split Wizard.
- Error Analysis Workspace.
- Statistical Analysis Wizard.
- DuckDB analytical workspace.
- Notebook interoperability.
- Dataset Cards / Model Cards / Publication Package.
- Model Monitoring / Drift.
- Typed Agent Tool Registry.
- Analysis Recipe / replay foundation.
- Agent Run Console.
- Analysis State Machine.
- Artifact-first architecture.
- PDF table extraction with explicit active-sheet activation.
- 100 professional Macro starters.
- Privacy-aware collaboration sharing.
- CPU/CPU+GPU/GPU selector.

## New high-level roadmap suggestions

The following are intentionally proposed as the next professional layer rather than silently added to the current release:

1. **Feature Set Challenge** — compare several feature sets under one locked validation protocol.
2. **Temporal Split Advisor** — automatically test whether a random split is inappropriate for time-dependent data.
3. **Robustness / Perturbation Suite** — noise, missingness, category perturbation and outlier sensitivity.
4. **Prediction Interval / Conformal Layer** — uncertainty around individual predictions where assumptions and sample size permit.
5. **Threshold Optimization Workspace** — classification threshold analysis with business/scientific cost functions.
6. **Data Slice Discovery** — find subgroups where model error is materially different, with minimum-support controls.
7. **Model Challenger Protocol** — require a simpler baseline/challenger before promotion.
8. **Feature Provenance Graph** — link each feature to source column, transformation, availability time and calculation recipe.
9. **Agent Stopping Criteria** — stop when evidence gain is below a declared threshold instead of using only a step counter.
10. **Tool Sandbox / Dry Run** — show the exact tool call, estimated cost and affected artifacts before execution.
11. **Agent Regression Suite** — replay benchmark datasets and compare agent plans, approvals, evidence completeness and failure modes across releases.
12. **Scientific Assumption Registry** — store explicit assumptions such as stationarity, independence, leakage exclusions and domain constraints.

These are particularly valuable for advanced users because they improve the **quality of the reasoning and evidence**, not merely the number of buttons.


## V18.6 additions

- **Compute crash hardening:** CPU+GPU is opportunistic and never formats missing VRAM as a number. NVIDIA hardware detection, PyTorch CUDA availability and runtime validation remain separate facts.
- **Robustness / Perturbation Agent:** bounded numeric jitter, random missingness and numeric clipping tests are executed against the locked test set and reported as sensitivity evidence.
- **Prediction uncertainty:** regression receives empirical residual interval diagnostics; classification receives probability-confidence/entropy diagnostics. These are explicitly labeled as uncertainty diagnostics, not guaranteed confidence intervals.
- **Tool Dry-Run:** every typed tool can be previewed without executing its handler. The preview shows required inputs, budgets, permissions and whether execution would be allowed.
- **Professional validation:** model evaluation now combines holdout performance, baseline comparison, repeated cross-validation, calibration where applicable, uncertainty, robustness perturbation, feature stability, temporal availability, counterfactual dependence and automatic diagnosis.
- **Marks drag/drop hardening:** Shift detection uses the application keyboard-modifier state rather than an unavailable `QDropEvent.keyboardModifiers()` method.
- **Plot formatting hardening:** Matplotlib color arguments are omitted when no valid color was selected; `None` is never passed as an explicit color.


### V18.6 agent execution policy

The Master Agent performs deterministic tool dry-runs during planning. A dry-run never executes a tool; it reports required inputs, observed data size, budget status, permissions and whether execution would be allowed.

Every approval proposal has an Agent Why evidence object and an Agent Self-Check bound to that Why object. Specialist-result approvals also bind their self-check to the corresponding specialist Why evidence.

The ML and DL evidence package includes bounded robustness perturbation tests (numeric 1% jitter, random 2% missingness and numeric 1–99% clipping) and prediction uncertainty diagnostics. These are sensitivity diagnostics and are not presented as guarantees of production performance or calibrated probability unless calibration has been validated.

The GUI remains unchanged: no new main-window panels, shelves, menus or layout regions are introduced for these backend capabilities.
