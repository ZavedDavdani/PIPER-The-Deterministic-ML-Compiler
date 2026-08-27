# PIPER — Complete Screenshot Gallery

This gallery documents the end-to-end visual workflow of the PIPER deterministic ML compiler across all execution, verification, and educational views.

---

## 1. Dataset Ingestion & Configuration

### Dataset Upload & Provider Configuration
Upload multi-format tabular datasets (CSV, TSV, Parquet, Excel, JSON) and configure planner providers (Google Gemini, OpenAI, or local Ollama) with instant dataset profiling.

![Dataset Upload & Configuration](screenshots/dataset-upload.png)

---

## 2. ML Education & Guided Walkthrough

### Student Mode — 14-Stage ML Learning Journey
A pedagogical walkthrough explaining every decision made during the run, including data profiling, imputation rationale, model architecture trade-offs, and controlled What-If sandboxes.

![Student Mode — 14-Stage ML Learning Journey](screenshots/student-mode.png)

### End-to-End Pipeline Visualization
Interactive visual pipeline flowchart tracing data from input ingestion through preprocessing, train/test partition, candidate training, evaluation, validation, and artifact deployment.

![End-to-End Pipeline Visualization](screenshots/pipeline-visualization.png)

---

## 3. Engineering & Decision Intelligence

### Decision Trace & Model Comparison
Full deterministic audit graph from proposal to final verdict, candidate model comparison bar charts, and step-by-step execution logs.

![Decision Trace & Model Comparison](screenshots/decision-trace.png)

### Dedicated Model Comparison
Side-by-side performance breakdown comparing Accuracy, Precision, Recall, F1, and ROC-AUC metrics against the deterministic selection gate.

![Model Comparison](screenshots/model-comparison.png)

---

## 4. Verification & Guardrails

### Baseline Gate
Evaluates candidate models against a zero-intelligence majority-class baseline to guarantee genuine predictive signal.

![Baseline Gate](screenshots/baseline-gate.png)

### Deterministic Safety Guardrails
Automated checks for data leakage, class imbalance, constant features, high cardinality, and suspicious evaluation metrics.

![Deterministic Safety Guardrails](screenshots/guardrails.png)

### Reproducibility Verification
Verifies bitwise reproducible execution with environment fingerprints, fixed seeds, and cross-run metric parity.

![Reproducibility Verification](screenshots/reproducibility.png)

---

## 5. Artifacts & Deployment

### Verified ML Artifact Bundle
Cryptographically verified bundle (\pipeline.joblib\, \pipeline.py\, \	raining_reproduction.ipynb\, \manifest.json\, \evidence.json\, \
equirements.txt\, \hashes.json\) passing strict \
p.array_equal\ parity gates.

![Verified ML Artifacts](screenshots/verified-artifacts.png)

### Test Flight (Unseen Data Inference)
Score new, unseen CSV or JSON records against the verified artifact in standalone mode without retraining.

![Test Flight Inference](screenshots/test-flight.png)
