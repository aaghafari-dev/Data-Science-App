# Data Science Studio Pro — V18.6

Data Science Studio Pro is a professional Windows/PyQt6 analytical environment designed for advanced data-science users. It combines a Tableau-like interactive analytical surface with governed LangGraph agents, professional ML/DL evaluation, leakage diagnostics, reproducibility, evidence management and report/presentation generation.

## GUI preservation

**The existing GUI and layout are preserved.** V18.6 changes backend logic and the behavior of existing controls. No main-window redesign is introduced.

## What V18.6 adds

### 1. Professional ML/DL evaluation with uncertainty and robustness testing

The application no longer treats one holdout R²/accuracy value as the complete model evaluation.

ML candidate models now receive:

- holdout metrics
- dummy baseline comparison
- repeated cross-validation
- CV mean/std/range
- train-vs-validation generalization gap
- bootstrap uncertainty where appropriate
- classification calibration diagnostics
- existing error analysis

DL records the equivalent held-out/baseline evidence plus hardware/runtime information and diagnostic evidence. The current neural implementation remains deliberately bounded for small-memory GPUs.

### 2. Professional GPU/CPU policy

The Compute selector remains:

- **CPU**
- **CPU+GPU**
- **GPU**

The new backend separates four questions that were previously conflated:

```text
Does Windows see an NVIDIA GPU?
          ↓
Does installed PyTorch contain CUDA?
          ↓
Can PyTorch initialize that GPU?
          ↓
Can a real CUDA operation execute safely?
```

It also records:

- GPU name
- VRAM total/free when `nvidia-smi` is available
- NVIDIA driver version
- PyTorch version
- CUDA build version
- compute capability when available
- compiled architecture list compatibility
- runtime smoke-test result

This means an NVIDIA GPU is not incorrectly described as “not found” merely because the Python environment contains a CPU-only PyTorch build or cannot initialize CUDA.

For an MX250 2 GB-class device, the application uses a conservative VRAM budget and can route large neural workloads to CPU rather than risking an out-of-memory crash.

### 3. Plan → Critic → Execute

The Master Agent now follows a governed pattern:

```text
Plan
  ↓
Critic
  ↓
Agent Self-Check
  ↓
Human Approval
  ↓
Execute
  ↓
Evidence + Result Self-Check
  ↓
Human Approval
```

Rejection causes revision/replanning. There is no automatic approval after a rejection.

### 4. Feature Stability Analysis

Repeated resampling and permutation importance estimate whether important features remain important across different samples. The output includes importance mean, standard deviation and positive-frequency.

### 5. Temporal Availability Matrix

For datasets containing a date/time field, the application produces an availability matrix showing the proportion of non-missing values for features over chronological periods.

Features that become available much later are flagged for temporal-leakage review.

### 6. Counterfactual Leakage Test

For suspicious features, the same model protocol is run after removing a feature. The resulting performance change measures model dependence and creates evidence for provenance review.

It is intentionally **not** described as causal proof of leakage.

### 7. Automatic Model Diagnosis

The diagnostic engine combines:

- generalization gaps
- calibration evidence
- feature stability
- temporal availability
- counterfactual dependence

into explicit diagnostic hypotheses and recommended next checks.

### 8. Agent Self-Check

Every approval request now has a machine-generated self-check covering:

- plan completeness
- evidence binding
- tool/action budget
- human approval requirement
- blocking reasons

The result is stored in the Evidence DAG.

### 9. Agent Why — first-class evidence

The new `agent_why` evidence object stores a concise, auditable explanation of an agent decision:

- decision
- rationale
- evidence references
- alternatives considered
- constraints

It does **not** attempt to store private chain-of-thought.

## Agent architecture

The core data-science architecture remains exactly:

```text
LangGraph
   ↓
Master Agent
   ├── ML Agent
   └── DL Agent
```

The Plot, Report, Presentation, Question and Table Creation workflows remain separate specialized LangGraph workflows. They do not change the core Master → ML/DL architecture.

## Advanced analytical flow

```text
Data
  ↓
Data Quality + Contract + Target Validation
  ↓
Leakage-aware Split
  ↓
Master Agent
  ↓
Plan → Critic → Self-Check → Human Approval
  ↓
ML / DL
  ↓
Professional Evaluation
  ├── baseline
  ├── repeated CV
  ├── uncertainty
  ├── calibration
  └── error analysis
  ↓
Advanced Diagnostics
  ├── feature stability
  ├── temporal availability
  ├── counterfactual leakage
  └── automatic diagnosis
  ↓
Agent Plot
  ↓
Evidence-bound Report
  ↓
Presentation / Publication / Sharing
```

## Existing professional functionality

V18.6 retains V18.4 functionality including:

- Tableau-like Rows/Columns/Marks semantics.
- Multiple Color/Text/Detail/Tooltip fields and single Size field.
- Data-driven AI Agent Plot with correlation, scatter, category, time-series and distribution views.
- Professional plot formatting and PNG/JPG/PDF export.
- Pie chart.
- Automatic target validation and A/A² relationship ambiguity detection.
- Data Contracts and Schema Drift.
- Leakage-aware split wizard.
- Error Analysis Workspace.
- Statistical Analysis Wizard.
- DuckDB workspace.
- Notebook export.
- Dataset Cards / Model Cards.
- Experiment Comparison.
- Model Monitoring / Drift.
- Analysis Recipe and replay foundation.
- Artifact registry and lineage.
- Evidence DAG.
- Human approval evidence.
- Agent Run Console.
- 100 Python/SQL macro starters.
- PDF table extraction into separate Loaded Sheets.
- Privacy-aware collaboration sharing.
- REST API / Docker / CI skeleton.

## Hardware guidance for the MX250 2 GB

The MX250 is a small Pascal-generation laptop GPU. The application therefore treats it as an optional accelerator rather than assuming it can execute every DL workload.

The important distinction is:

```text
GPU physically present
        ≠
PyTorch CUDA build installed
        ≠
CUDA initialized
        ≠
CUDA kernel execution validated
```

The application checks these conditions separately.

Current PyTorch packaging is also relevant to legacy Pascal GPUs. PyTorch's current documentation and release notices indicate that published CUDA wheels are changing over time and that Pascal support requires attention to the selected PyTorch/CUDA build. Therefore the application records the exact PyTorch version, CUDA build and device architecture instead of assuming that “CUDA installed” means that the selected Python environment can use the GPU.

## Installation

```bash
python -m pip install -r requirements.txt
python main.py
```

Optional integrations:

```bash
python -m pip install -r requirements-optional.txt
```

## Validation

Run:

```bash
python -m compileall -q .
pytest -q
```

The V18.6 development validation completed with:

**28 passed, 1 skipped**

The skipped test is the optional LangGraph/LangChain integration test when those dependencies are unavailable in the validation environment.

Live PyQt6 interaction was not claimed as tested in the headless validation environment.

## Important design principle

The application is **evidence-first**:

> data → quality → contract → target/leakage controls → governed agent plan → human approval → execution → professional evaluation → diagnostics → visualization → report → reproducible sharing

The objective is not simply to automate model training. The objective is to give an advanced data-science user a transparent, reproducible and auditable analytical workflow.

## Next professional extensions

Recommended next-layer capabilities:

1. Feature Set Challenge under one locked validation protocol.
2. Temporal Split Advisor.
3. Robustness/Perturbation Suite.
4. Prediction intervals / conformal prediction where justified.
5. Classification Threshold Optimization.
6. Data Slice Discovery with minimum-support controls.
7. Model Challenger Protocol.
8. Feature Provenance Graph.
9. Evidence-based Agent Stopping Criteria.
10. Tool Sandbox / Dry Run with cost estimates.
11. Agent Regression Suite.
12. Scientific Assumption Registry.

These are deliberately framed as extensions because the current release already has a strong governed agent foundation.


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
