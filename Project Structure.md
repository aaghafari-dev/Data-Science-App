# Data Science Studio Pro — Project Structure (V18.3)

## 1. Product principle

Data Science Studio Pro is a professional Windows/Python data-analysis environment combining a Tableau-style analytical surface with governed LangGraph agents, reproducibility, statistical analysis and practical MLOps.

**GUI rule:** the approved main GUI geometry and layout are preserved. The central plot, Rows/Columns shelves, Marks panel, right-side Data Management/Filters/Extra Insight docks and existing toolbar placement are not redesigned. New behavior is implemented behind the existing controls and through menus/dialogs/services.

## 2. Interaction architecture

### Marks
Five existing controls remain in their current location. Each is a real field assignment control:

- Color — categorical grouping or quantitative color encoding.
- Size — quantitative visual size.
- Text — visible annotation.
- Detail — additional grouping/granularity.
- Tooltip — hover-only context.

Each selector has two valid input paths: live dropdown and drag/drop from Data Management. The selector synchronizes from the active DataFrame when opened, so it cannot show an empty/stale field list after a sheet switch.

### Filters
The Filters dock is a real drop target. Numeric, categorical and datetime columns are accepted and converted into persistent filter specifications.

### PDF tables
File → Open PDF extracts tables page-by-page. A selection dialog shows every detected table, page and dimensions, with Select All / Clear All and preview. Selected tables become named sheets in Loaded Sheets. The first imported sheet is explicitly selected so the active working table is unambiguous.

### Macro workspace
Macro → Python / SQL opens a multi-run workspace inspired by the V16 implementation: starter library + editor + execution output/history on the same page. Run Current does not close the dialog; users can edit and execute several commands sequentially.

## 3. Core data model

Stage 1 is deliberately limited to the four foundational analytical dtypes:

- `int64`
- `float64`
- `object`
- `datetime`

Semantic roles such as Identifier, Target, Measure, Ordinal and Geographic are a second-stage layer. This prevents the basic pandas dtype engine from being mixed with higher-level interpretation.

All-missing predictor columns are removed before sklearn imputation, eliminating the warning/error path caused by columns such as `Daltonic` that contain no observed values.

## 4. Professional services

```text
services/
├── agent_memory.py              # bounded role-specific working memory
├── agent_tools.py               # typed/versioned agent tool registry
├── analysis_recipe.py           # reproducible action recipe
├── analysis_state.py            # explicit project lifecycle state machine
├── agent_console.py             # agent node/status/evidence event log
├── artifacts.py                 # artifact-first registry and lineage
├── target_feature_selection.py  # target/feature candidate analysis
├── compute_backend.py           # CPU / CPU+GPU / GPU policy and detection
├── pdf_tables.py                # PDF table extraction and identity
├── data_contracts.py            # schema contracts + drift
├── split_wizard.py              # leakage-aware splits
├── error_analysis.py            # model error diagnostics
├── statistical_analysis.py      # statistical workflow engine
├── duckdb_workspace.py          # SQL analytical workspace
├── notebook_export.py           # reproducible notebook export
├── monitoring.py                # data/model drift diagnostics
├── publication_package.py       # publication bundle
├── sharing.py                   # privacy-aware collaboration bundle
├── macro_library.py             # 100 Python/SQL starter commands
├── model_preprocessing.py       # dtype-safe sklearn preprocessing
└── ...
```

## 5. Agent architecture

```text
                         LangGraph
                            │
                  ┌─────────┴─────────┐
                  │                   │
             Master Agent       Specialist Agents
                  │                   │
          ┌───────┴───────┐     ┌────┴─────┐
          ▼               ▼     ▼          ▼
       ML Agent         DL Agent Plot     Report
          │               │                 │
          └───────┬───────┘                 ▼
                  ▼                    Presentation
             Human Gates               (Streamlit)
```

LangGraph is the orchestration engine: nodes, transitions, state and checkpoints/memory. Benchmarking is a separate evaluation layer and never becomes a second orchestrator.

## 6. Agent memory model

Each agent has bounded role-specific working memory. Memory stores reusable context, not secrets.

- **Master Agent:** prior route decisions, approved/rejected milestones, target-analysis context and run status.
- **ML Agent:** preprocessing decisions, task type, model/evaluation context and previous model-analysis summaries.
- **DL Agent:** compute backend, architecture/training context and prior DL results.
- **Question → Analysis Agent:** questions, target candidates, feature candidates and warnings.
- **Table Creation Agent:** source/method/success summaries; API keys are excluded.
- **Plot Agent:** model-result structures and plotting choices.
- **Report Agent:** evidence keys and report-generation context.
- **Presentation Agent:** report/presentation generation context and prior output metadata.

Memory is bounded to avoid unbounded context growth and is deliberately separated by agent role.

## 7. Target and feature selection

The Question → Analysis workflow first proposes candidates using:

- column-name signals
- dtype
- cardinality
- missingness
- constant/invalid detection
- identifier-like patterns
- possible future/post-outcome names
- numeric target correlation signals

The system **does not silently choose a target for a consequential model run**. It presents candidate targets/features to the human, who can approve or change them. The Master Agent then receives the approved target and feature context.

## 8. Typed Agent Tool Registry

`TypedAgentToolRegistry` is an explicit allow-list. Every tool declares:

- name
- version
- description
- input schema
- capabilities
- handler

Agents cannot invent arbitrary tool calls. This provides a controlled extension point without creating an unrestricted plugin execution layer.

## 9. Analysis Recipe

Every important action can be recorded as a reproducible recipe step:

```text
Load → Quality → Contract → Filters → Transformations → Split → Model → Evaluate → Plot → Report → Share
```

Each step can reference artifact and evidence IDs.

## 10. Analysis State Machine

```text
DATA_LOADED
  ↓
QUALITY_CHECKED
  ↓
CONTRACT_VALIDATED
  ↓
ANALYSIS_READY
  ↓
MODEL_READY
  ↓
MODEL_VALIDATED
  ↓
REPORT_READY
  ↓
PRESENTATION_READY
  ↓
SHAREABLE
  ↓
MONITORED
```

Invalid transitions are rejected rather than silently accepted.

## 11. Artifact-first architecture

Dataset, model, plot, report, presentation, recipe and governance objects become explicit artifacts with:

- artifact ID
- version
- parent artifacts
- metadata
- creation time

This allows evidence and reproducibility to reference concrete objects rather than loose strings.

## 12. Compute policy

The existing toolbar position is preserved. A compact compute selector provides:

- CPU
- CPU+GPU
- GPU

PyTorch CUDA is detected at runtime. If GPU is requested but no CUDA-capable GPU is found, the user receives an explicit message and the application safely falls back to CPU. CPU+GPU uses CPU-oriented ML components and GPU-capable DL components where available.

## 13. PDF table workflow

```text
Open PDF
   ↓
Extract page/table candidates
   ↓
Select All / Clear All / Preview
   ↓
Select desired tables
   ↓
Loaded Sheets
   ↓
Select active sheet
   ↓
Data Management / Rows / Columns / Marks / Filters
```

Every imported sheet retains source PDF, page and table-index metadata.

## 14. Macro workflow

The 100-command library contains 50 Python/pandas/NumPy/SciPy commands and 50 SQL/DuckDB commands. The editor supports repeated execution without closing the dialog. Output and errors remain on the same page, enabling iterative data exploration.

## 15. Agent Question → Analysis

The Question Agent interprets a natural-language data-science question, proposes target and feature candidates, identifies warnings and builds an analysis plan. Human review is required before the plan is handed to the Master Agent.

## 16. Validation

The project is tested with `pytest` and Python compilation. Optional LangGraph/LLM integration tests may be skipped when optional dependencies are unavailable.
