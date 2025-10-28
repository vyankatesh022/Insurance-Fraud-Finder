# 🧠 Insurance-Fraud-Finder

A complete **end-to-end MLOps project** that automates preprocessing, model training, evaluation and experiment tracking — built using **DVC**, **MLflow**, and **Python**.

---

## 🚀 Features

✅ **Data Versioning** – DVC manages data lineage and reproducibility.  
✅ **Experiment Tracking** – MLflow logs model metrics, parameters, and artifacts.  
✅ **Multiple Models** – Supports Random Forest and Deep Learning models.  
✅ **Config-Driven** – All hyperparameters and file paths managed via `params.yaml`.  
✅ **Logging** – Centralized logging in `logs/app.log`.  
✅ **Local-Only Setup** – All operations are performed locally; data and artifacts stay private.  

---

## 📦 Folder Structure

```bash
Insurance-Fraud-Finder:.
├── .dvc/                           # DVC internal configuration and cache
│   ├── config
│   └── cache/
│
├── data/
│   ├── raw/                        # Unprocessed data (ignored by Git)
│   │   └── Fraud_data.csv
│   └── processed/                  # Cleaned and feature-engineered data
│       └── fraud_data_processed.csv
│
├── logs/
│   └── app.log                     # Unified pipeline logs
│
├── metrics/
│   ├── rf_metrics.json             # Random Forest evaluation metrics
│   └── dl_evaluation_report.json   # Deep Learning evaluation report
│
├── mlruns/                         # MLflow tracking data (ignored by Git)
├── mlartifacts/                    # MLflow artifact storage (ignored by Git)
│
├── models/
│   ├── rf_model.pkl                # Random Forest model
│   └── dl_model.pkl                # Deep Learning model
│
├── src/                            # Source code for all pipeline stages
│   ├── data_preprocessing.py
│   ├── train_rf.py
│   ├── evaluate_rf.py
│   ├── train_dl.py
│   ├── evaluate_dl.py
│   ├── mlflow_tracking.py
│   ├── logger_utils.py
│   └── __pycache__/
│
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # Auto-generated dependency tracking
├── params.yaml                     # Config file for parameters and paths
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignore data, logs, and artifacts
└── README.md                       # Project documentation (this file)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/vyankatesh022/Insurance-Fraud-Finder.git
cd Insurance-Fraud-Finder
```

### 2️⃣ Set Up Python Environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
pip install -r requirement.txt
```

### 3️⃣ Configure DVC (optional if you use DVC)

```bash
dvc init
```

---

## 🧪 Running the Pipeline Locally

### Preprocess → Train → Evaluate

```bash
dvc repro
```

### Or run manually:

```bash
python src/data_preprocessing.py
python src/train_rf.py
python src/evaluate_rf.py
python src/train_dl.py
python src/evaluate_dl.py
```

---

## 📊 Experiment Tracking with MLflow

### Launch MLflow locally

```bash
mlflow ui
```

Open [DagsHub](https://dagshub.com/vyankatesh/Insurance-Fraud-Finder.mlflow) to view your experiment logs.

MLflow automatically logs:

* Model parameters
* Training and validation metrics
* Saved model artifacts

---

## 🧱 params.yaml Example

```yaml
data:
  raw_file: "data/raw/Fraud_data.csv"
  processed_file: "data/processed/fraud_data_processed.csv"

rf_model:
  n_estimators: 100
  max_depth: 10
  test_size: 0.2
  random_state: 42

dl_model:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  units: 64
  dropout: 0.3
  test_size: 0.2
  random_state: 42

logging:
  file: "logs/app.log"
  level: "INFO"
```

---

## 🧠 Logging Example

All modules log events to `logs/app.log`:

```
data_preprocessing - INFO - Data cleaned successfully.
train_rf - INFO - Model trained and saved to models/rf_model.pkl
evaluate_dl - INFO - Validation accuracy: 0.86
```

---

## 🧰 Useful Commands

| Task                    | Command                                            |
| ----------------------- | -------------------------------------------------- |
| Run full pipeline       | `dvc repro`                                        |
| View pipeline graph     | `dvc dag`                                          |
| Launch MLflow dashboard | `mlflow ui`                                        |
| Clean generated files   | `rm -rf data/processed models metrics logs mlruns` |
| Check log output        | `tail -f logs/app.log`                             |

---

## 🧩 Future Improvements

* [ ] Add automatic model versioning via MLflow Registry
* [ ] Add Slack/email notifications for CI failures
* [ ] Integrate unit tests for preprocessing and model metrics
* [ ] Extend support for XGBoost and LightGBM

---

## 👨‍💻 Author

**Vyankatesh**  
🌐 [GitHub Profile](https://github.com/vyankatesh022)

---