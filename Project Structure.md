# Project Structure

## 1. Purpose

Data Science Studio Pro is a professional Windows desktop data-analysis environment designed around:

- evidence-first analysis
- human-controlled AI agents
- reproducible analytical workflows
- Tableau-like visual exploration
- professional supervised ML/DL evaluation
- uncertainty and robustness analysis
- governed reports and presentations
- safe CPU/GPU execution

The existing GUI geometry and primary layout are intentionally preserved.

## 2.  Architecture

```text
GUI
 ├── Data Management
 ├── Rows / Columns
 ├── Marks
 ├── Plot Area
 ├── Filters / Extra Insight
 ├── Sheets / Stories
 ├── Compute selector
 ├── Accessibility / Voice
 └── Professional Help Center

LangGraph
 └── Master Agent
      ├── PLAN
      ├── CRITIC
      ├── SELF-CHECK
      ├── HUMAN APPROVAL
      ├── VALIDATION
      ├── PREPROCESSING REVIEW
      ├── MODEL SELECTION REVIEW
      ├── MODEL EXECUTION
      ├── EVALUATION REVIEW
      ├── ROBUSTNESS REVIEW
      ├── DIAGNOSIS REVIEW
      ├── INDEPENDENT VERIFICATION
      └── STOP EVALUATOR
             │
             ├── ML Agent
             └── DL Agent
```

Every governed Master-Agent stage requires explicit user approval before the graph can advance.

## 3. Important source files

```text
main.py
agent/
├── graph_ds.py                 # Master Agent and stage-by-stage approval workflow
├── graph_ml.py                 # Professional ML agent
├── graph_dl.py                 # Professional DL agent
├── graph_plot.py               # Data-driven Plot Agent
├── graph_report.py             # Evidence-bound professional report + embedded plots
├── graph_presentation.py       # Streamlit presentation generator
├── graph_question_analysis.py
├── graph_table_creation.py
├── local_llm.py
├── prompts.py
└── voice_layer.py

core/
├── data_engine.py
├── viz_engine.py               # Shelf planning / aggregation / Plotly rendering
├── tableau_views.py
├── sheet_manager.py
└── theme_manager.py

services/
├── supervised_analysis.py      # Classification + Regression workspaces
├── evaluation_protocol.py
├── agent_quality.py
├── result_verification.py
├── evidence_validation.py
├── tool_preflight.py
├── agent_tools.py
├── analysis_cache.py
├── feature_set_analysis.py
├── feature_provenance.py
├── data_slice_analysis.py
├── target_feature_selection.py
├── compute_backend.py
├── model_registry.py
├── experiment_registry.py
├── monitoring.py
├── analysis_recipe.py
├── artifacts.py
├── governance.py
├── split_wizard.py
├── error_analysis.py
├── model_preprocessing.py
├── statistical_analysis.py
├── data_quality.py
├── data_contracts.py
├── publication_package.py
└── sharing.py
```

## 4. Classification Analysis

The dedicated Classification Analysis workspace implements:

### Preprocessing

- missing-value imputation
- categorical encoding with `OneHotEncoder(handle_unknown="ignore")`
- fold-local feature scaling
- optional PCA
- preprocessing inside the model pipeline

### Methods

- Logistic Regression
- Support Vector Machine
- Random Forest
- Decision Tree
- KNN
- Naive Bayes
- Feed-forward Neural Network
- optional XGBoost when installed
- optional LightGBM when installed

### Model selection

- stratified k-fold CV
- GridSearchCV / RandomizedSearchCV style hyperparameter optimization
- training-only selection
- independent locked test partition

### Evaluation

- accuracy
- balanced accuracy
- precision macro/weighted
- recall macro/weighted
- F1 macro/weighted
- confusion matrix
- ROC-AUC
- PR-AUC
- log loss
- Brier score
- calibration diagnostics

## 5. Regression Analysis

Methods:

- OLS
- Ridge
- Lasso
- Elastic Net
- Polynomial Regression
- SVR
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting
- Neural Network
- optional XGBoost
- optional LightGBM

Metrics:

- MAE
- RMSE
- R²
- median absolute error
- mean bias error
- MAPE when mathematically meaningful

The final test partition remains locked for the final generalisation estimate.

## 6. Master Agent governance

The Master Agent does not silently move from planning to execution.

```text
PLAN
 ↓ approval
VALIDATION
 ↓ approval
PREPROCESSING
 ↓ approval
MODEL SELECTION
 ↓ approval
MODEL EXECUTION
 ↓ approval
EVALUATION
 ↓ approval
ROBUSTNESS
 ↓ approval
DIAGNOSIS
 ↓ approval
VERIFY
 ↓ approval
STOP
```

Rejection returns the current stage to a revised proposal. Abort stops the run safely.

## 7. Agent evidence

Agent Why and Agent Self-Check are first-class evidence objects.

Evidence is bound to:

- decision
- rationale
- alternatives
- constraints
- evidence references
- approval state

Private chain-of-thought is not stored.

## 8. AI Agent Report

The Report Agent consumes:

1. Agent Data Scientist evidence
2. Agent Plot results
3. LLM output
4. governance/evidence information

The report contains:

- executive summary
- analytical objective
- dataset/governance
- methodology
- validation
- model comparison
- uncertainty
- robustness
- diagnosis
- visual findings
- human oversight
- limitations
- reproducibility
- evidence appendix

Plotly visualizations are embedded as report figures when the Plotly image backend is available.

## 9. AI Agent Presentation

Presentation generation is JSON/msgpack-safe. DataFrames are converted into bounded evidence summaries before being passed to LangGraph/checkpointing.

The generated Streamlit artifact consumes report text, model evidence and serialized Plotly figures.

## 10. Tableau-like visualization behavior

### Rows / Columns

Right-click a field to select:

- Automatic
- Sum
- Average
- Count
- Minimum
- Maximum
- Median
- Count Distinct
- Remove

### Marks

Color, Size, Text, Detail and Tooltip have actual field-selection state.

Clear removes the selected field state rather than merely changing the combo-box index.

Categorical Color generates a legend. Numeric Color generates a continuous color encoding.

## 11. Sheets and Stories

Sheets are independent view snapshots containing:

- Rows
- Columns
- chart type
- Marks
- aggregation
- data snapshot

Stories allow the user to:

- select plotted sheets
- order sheets
- add/remove story points
- add explanatory text
- preview the story
- export PNG
- export PDF

The main application layout is not redesigned.

## 12. Compute architecture

```text
Hardware detection
      ↓
Driver detection
      ↓
PyTorch CUDA build
      ↓
Architecture compatibility
      ↓
CUDA runtime validation
      ↓
VRAM/resource preflight
      ↓
Execution policy
```

CPU+GPU is opportunistic. GPU mode is strict. Low-VRAM devices are protected by resource limits.

## 13. Tool architecture

Every important analytical tool should expose:

- typed inputs
- version
- capabilities
- resource budget
- evidence requirement
- approval requirement
- deterministic/cacheable status
- dry-run information

Tool Dry-Run never executes the tool.

## 14. Testing

The test suite covers:

- core services
- visualization behavior
- governance
- agent quality
- V20 supervised-method contracts
- Master-Agent approval staging
- presentation serialization safeguards

Optional LangGraph/LangChain tests are skipped when the optional dependency is unavailable in the validation environment.
