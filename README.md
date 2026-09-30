# Data Science Studio Pro 
Professional desktop data-analysis software for Windows.

## 

strengthens the application around five principles:

1. **Trust** — do not accept a result merely because code executed successfully.
2. **Evidence** — important conclusions must be traceable to analytical artifacts.
3. **Method Selection** — choose methods according to the question, data structure and validation requirements.
4. **Efficiency** — dry-run tools, estimate resources and avoid unnecessary recomputation.
5. **Human Control** — every governed Agent Data Scientist stage requires explicit approval.

The existing GUI and layout are preserved.

## Major  capabilities

### Agent Data Scientist

LangGraph now coordinates:

```text
Plan
 → Critic
 → Self-Check
 → Human Approval
 → Validation
 → Human Approval
 → Preprocessing Review
 → Human Approval
 → Model Selection Review
 → Human Approval
 → Model Execution
 → Human Approval
 → Evaluation Review
 → Human Approval
 → Robustness Review
 → Human Approval
 → Diagnosis Review
 → Human Approval
 → Independent Verification
 → Human Approval
 → Stop Evaluator
```

The Master Agent still contains exactly two internal specialist agents:

- ML Agent
- DL Agent

A rejection revises the current stage. Abort terminates safely.

### Professional classification analysis

The dedicated workspace implements:

- missing-value imputation
- categorical encoding
- fold-local scaling
- optional PCA
- k-fold validation
- hyperparameter search
- locked final test set
- Logistic Regression
- SVM
- Random Forest
- Decision Tree
- KNN
- Naive Bayes
- Neural Network
- optional XGBoost
- optional LightGBM
- confusion matrix
- accuracy
- balanced accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- log loss
- Brier score
- calibration evidence

### Professional regression analysis

Methods:

- OLS
- Ridge
- Lasso
- Elastic Net
- Polynomial Regression
- SVR
- Decision Tree
- Random Forest
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
- MAPE where meaningful

### AI Agent Report

The Report Agent combines Agent Data Scientist evidence, Agent Plot visualizations and the selected LLM.

The report includes professional analytical explanations and embeds Plotly visualizations into the PDF when the image-rendering backend is installed.

### AI Agent Presentation

DataFrames are converted into bounded JSON/msgpack-safe evidence summaries before presentation generation. This prevents the previous:

```text
Type is not msgpack serializable: DataFrame
```

failure.

### Tableau-like Rows / Columns aggregation

Right-click a field on Rows or Columns to choose:

- Automatic
- Sum
- Average
- Count
- Minimum
- Maximum
- Median
- Count Distinct
- Remove

### Tableau-like Marks

Clear now clears the actual internal field selection.

Categorical Color creates a legend showing the meaning of each color.

### Sheets and Stories

The professional workspace supports:

- New Sheet
- Duplicate
- Rename
- Delete
- activate a sheet
- select plotted sheets for a Story
- reorder story sheets
- explanatory story text
- Story preview
- Story PNG export
- Story PDF export

### Professional Help Center

Help now includes:

- Quick Start
- Rows / Columns / Aggregation
- Marks
- Classification Analysis
- Regression Analysis
- Agent Data Scientist
- AI Agent Report
- Sheets and Stories
- Voice / Accessibility
- CPU / GPU
- Troubleshooting
- Reproducibility

### Voice

Voice is opt-in. Accessibility → Start Voice Command opens a dedicated control panel.

Only unambiguous commands can control an approval gate:

- approve
- reject
- pause
- abort

Ambiguous speech never approves a step.

### Compute

The compute policy distinguishes:

- physical NVIDIA hardware
- driver availability
- PyTorch CUDA build
- architecture compatibility
- CUDA runtime initialization
- VRAM/resource limits

CPU+GPU is opportunistic. GPU mode is strict. Low-VRAM devices such as a 2 GB MX250 are protected by bounded execution policies.

## Reproducibility

The application records, where applicable:

- dataset fingerprint
- data contract
- target/feature decisions
- validation protocol
- model configuration
- random seeds
- agent decisions
- approval evidence
- artifacts
- evidence DAG
- Analysis Recipe
- model/experiment registry information

The final test partition is not used for model selection or hyperparameter tuning.

## Running

```powershell
py -3.11 .\main.py
```



Skipped tests are optional integrations when LangGraph/LangChain is not installed in the lightweight validation environment.

## Optional packages

For extended functionality consider:

- `xgboost`
- `lightgbm`
- `kaleido` — Plotly static-image export for professional PDF reports
- `mlflow`
- `optuna`
- `shap`
- `statsmodels`
- `ydata-profiling`

## Design policy

The application is intended for a high-level data-analysis user. Automation must not replace methodological judgment. The software therefore favors:

- explicit validation
- independent verification
- uncertainty
- robustness
- evidence traceability
- human approval
- reproducibility

over opaque automatic conclusions.

## Recommended next development priorities

1. Conformal prediction with explicit coverage validation.
2. Group-aware and temporal nested cross-validation in the dedicated supervised workspaces.
3. Data-slice discovery with statistical multiplicity controls.
4. Feature provenance graphs from raw source through transformed model input.
5. Automatic residual diagnostics and influence analysis.
6. Experiment-level cost/performance/uncertainty frontiers.
7. Agent stopping criteria based on information gain rather than only step count.
8. Parallel execution of independent diagnostics under a governed resource budget.
9. Deployment simulation: batch inference, latency, memory and drift rehearsal.
10. Scientific report claim-to-evidence verification before publication export.
