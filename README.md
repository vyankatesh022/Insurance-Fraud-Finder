# 🧠 Insurance-Fraud-Finder

![Python Version](https://img.shields.io/badge/python-3.9-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-orange)
![CI/CD](https://github.com/vyankatesh022/Insurance-Fraud-Finder/actions/workflows/ci.yml/badge.svg)

## 📖 Overview

A complete **end-to-end MLOps project** that automates preprocessing, model training, evaluation and experiment tracking — built using **DVC**, **MLflow**, and **Python**. The pipeline seamlessly detects insurance fraud by maintaining data lineage, version control, and model metrics for full reproducibility.

## 🏗️ Architecture & MLOps Pipeline

This repository leverages modern MLOps tools to streamline the machine learning lifecycle:
- **DVC (Data Version Control):** Tracks changes to large datasets and binary models. It automatically builds a dependency graph of data transformations.
- **MLflow:** Acts as a centralized tracking server to record hyperparameters, evaluation metrics, and artifacts across multiple models and training runs.
- **GitHub Actions (CI/CD):** Automatically lints and tests code upon every push or pull request to ensure high code quality.

## 🚀 Key Features

- ✅ **Data Versioning:** DVC manages data lineage and reproducibility.
- ✅ **Experiment Tracking:** MLflow logs model metrics, parameters, and artifacts locally or on DagsHub.
- ✅ **Multiple Models:** Supports robust baseline models like **Random Forest**, as well as deep learning implementations via **Keras** and **PyTorch**.
- ✅ **Config-Driven:** All hyperparameters and file paths are strictly managed via a centralized `params.yaml` file.
- ✅ **Local-Only Setup:** All core operations can be performed locally; keeping sensitive data and artifacts completely private.
- ✅ **CI/CD Pipeline:** Automated testing and formatting checks via GitHub Actions.

## 🏁 Getting Started

### Prerequisites
- Python 3.9+
- Git
- DVC (installed via pip requirements)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vyankatesh022/Insurance-Fraud-Finder.git
   cd Insurance-Fraud-Finder
   ```

2. **Set up Python Environment:**
   ```bash
   python -m venv .venv
   # Activate on Linux/Mac:
   source .venv/bin/activate
   # Activate on Windows:
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

4. **Initialize DVC (If required for local tracking):**
   ```bash
   dvc init
   ```

## 🛠️ Usage

### Configuring Parameters

Before running the pipeline, ensure your settings in `params.yaml` are correct. 

<details>
<summary><b>Click to view a sample `params.yaml` configuration</b></summary>

```yaml
data:
  raw_path: "data/raw/Fraud_data.csv"
  processed_file: "data/processed/fraud_data_processed.csv"

rf_model:
  test_size: 0.4
  random_state: 30
  cv_folds: 3

dl_model:
  test_size: 0.4
  epochs: 100
  batch_size: 32

pytorch_model:
  test_size: 0.4
  epochs: 100
  learning_rate: 0.001

logging:
  file: "logs/app.log"
  level: "INFO"
```
</details>

### Running the Pipeline Locally

You can run the full end-to-end pipeline (Preprocess → Train → Evaluate) using DVC:
```bash
dvc repro
```

**Alternatively, run stages manually:**
```bash
python src/data_preprocessing.py
python src/train_rf.py
python src/evaluate_rf.py
python src/train_dl.py
python src/evaluate_dl.py
python src/train_pytorch.py
python src/evaluate_pytorch.py
```

## 📊 Experiment Tracking with MLflow

MLflow is integrated natively to track model parameters, training metrics, and artifacts automatically.

1. **Launch MLflow UI locally:**
   ```bash
   mlflow ui
   ```
2. Navigate to `http://127.0.0.1:5000` to view the MLflow dashboard.

*Note: You can also open [DagsHub](https://dagshub.com/vyankatesh/Insurance-Fraud-Finder.mlflow) to view your remote experiment logs.*

## 📦 Repository Structure

```text
Insurance-Fraud-Finder/
├── .github/workflows/              # GitHub Actions CI/CD workflows
├── .dvc/                           # DVC internal configuration and cache
├── data/                           # Ignored by Git (tracked via DVC)
│   ├── raw/                        # Unprocessed source data
│   └── processed/                  # Cleaned and feature-engineered data
├── logs/                           # Unified application logs (e.g., app.log)
├── metrics/                        # Model evaluation metrics (JSON formats)
├── models/                         # Serialized models (pkl files)
├── src/                            # Source code for pipeline stages
│   ├── data_preprocessing.py
│   ├── train_*.py                  # Training scripts (rf, dl, pytorch)
│   ├── evaluate_*.py               # Evaluation scripts
│   ├── logger_utils.py             # Logging configuration module
│   └── mlflow_tracking.py          # MLflow integration utilities
├── dvc.yaml                        # DVC pipeline definition graph
├── dvc.lock                        # Auto-generated dependency state tracking
├── params.yaml                     # Global configurations & hyperparameters
├── requirement.txt                 # Python dependencies list
└── README.md                       # Project documentation
```

## 💻 Development & CI/CD

### Logging
All internal modules log structured events centrally to `logs/app.log`. 
```text
data_preprocessing - INFO - Data cleaned successfully.
train_rf - INFO - Model trained and saved to models/rf_model.pkl
evaluate_dl - INFO - Validation accuracy: 0.86
train_pytorch - INFO - Epoch [100/100], Loss: 0.3541
```

### Useful Commands

| Task | Command |
|------|---------|
| Run full pipeline | `dvc repro` |
| View pipeline graph | `dvc dag` |
| Launch MLflow dashboard | `mlflow ui` |
| Clean generated files | `rm -rf data/processed models metrics logs mlruns` |
| Check log output | `tail -f logs/app.log` |

## 🧩 Future Improvements

* [ ] Add automatic model versioning via MLflow Registry
* [ ] Add Slack/email notifications for CI failures
* [ ] Integrate unit tests for preprocessing and model metrics
* [ ] Extend support for XGBoost and LightGBM

## 👨‍💻 Author

**Vyankatesh**  
🌐 [GitHub Profile](https://github.com/vyankatesh022)