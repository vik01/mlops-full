# Heart Disease Prediction – MLOps Pipelin

**Author:** Vikram Bhatt, Teresa Alvarez, Alessandro Cristofolini, Matteo Khoueiri, Dominique Robson, & Pengchong Zhao  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Session 1 (Initialization)

---

## 1. Getting Started

Follow these steps to set up the project environment locally. This assumes you
already have installed python and pip.

### 1.1 Create a Python virtual environment

```bash
python -m venv venv
```

### 1.2 Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
```

To verify the installation:
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, pytest; print('All packages installed successfully!')"
```

### 1.4 Deactivate when done

```bash
deactivate
```

---

## 2. Business Objective

* **The Goal:** Reliably predict the presence of heart disease using patients' clinical data, enabling early screening and risk stratification. The model supports clinical decision-making by maximizing true positive detection while minimizing false positives — reducing missed diagnoses and unnecessary follow-up procedures.

* **The User:** Healthcare professionals and screening platforms receive predictions via a web-accessible API endpoint. Users can submit patient clinical features and receive a heart disease risk prediction in real time.

---

## 3. Success Metrics

* **Business KPI (The "Why"):**
  > Improve early detection of heart disease by correctly identifying ≥87% of positive cases (recall) while keeping the false positive rate below 12%, reducing both missed diagnoses and unnecessary referrals.

* **Technical Metric (The "How"):**
  > - Logistic Regression test accuracy ≥ 88%
  > - ROC-AUC ≥ 0.95
  > - F1-score ≥ 0.86 on the test set

* **Acceptance Criteria:**
  > The primary model (Logistic Regression) must outperform the Decision Tree baseline and demonstrate consistent performance across precision, recall, specificity, and F1-score.

---

## 4. The Data

* **Source:** Synthetic heart disease dataset sourced from OpenML / Kaggle (train.csv and test.csv). The dataset contains 630,000 observations and 13 clinical features.
* **Target Variable:** `Heart Disease` — binary classification (Presence / Absence).
* **Features:**
  | Feature | Type | Description |
  |---|---|---|
  | Age | Numeric | Patient age |
  | Sex | Categorical | Patient sex (binary) |
  | Chest pain type | Categorical | Type of chest pain (1–4) |
  | BP | Numeric | Resting blood pressure |
  | Cholesterol | Numeric | Serum cholesterol |
  | FBS over 120 | Categorical | Fasting blood sugar > 120 mg/dl (binary) |
  | EKG results | Categorical | Resting electrocardiographic results |
  | Max HR | Numeric | Maximum heart rate achieved |
  | Exercise angina | Categorical | Exercise-induced angina (binary) |
  | ST depression | Numeric | ST depression induced by exercise |
  | Slope of ST | Categorical | Slope of the peak exercise ST segment |
  | Number of vessels fluro | Categorical | Number of major vessels colored by fluoroscopy |
  | Thallium | Categorical | Thallium stress test result |

* **Sensitive Info:** No PII (emails, credit cards, etc.) present. The dataset is synthetically generated.
  > ⚠️ **WARNING:** Raw data files should still NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.

---

## 5. Models

Three modeling approaches were applied and compared:

### 5.1 K-Means Clustering (Unsupervised)
- Used for patient segmentation and risk profiling (not direct prediction).
- Features were scaled with `StandardScaler`; optimal k=2 selected via Elbow Method and Silhouette Score.
- **Cluster 0 (High-risk):** Older, predominantly male patients with lower max heart rate, higher ST depression, frequent exercise-induced angina, abnormal thallium results, and more affected vessels.
- **Cluster 1 (Lower-risk):** Younger patients with balanced gender distribution, minimal ST depression, rare exercise-induced angina, and mostly normal clinical presentation.
- Ex-post label mapping accuracy: 87.4%

### 5.2 Logistic Regression (Primary Model)
- Binary classification with L2 regularization (`lbfgs` solver, `max_iter=2000`).
- Optimal threshold selected via Youden's J statistic on the ROC curve.
- **ROC-AUC: 0.9502**
- **Top 5 predictors:** Exercise angina, Sex, Chest pain type, Number of vessels fluro, Slope of ST.

### 5.3 Decision Tree (Baseline)
- `max_depth=4`, `min_samples_leaf=30` for controlled complexity.
- Serves as an interpretable reference point, not the primary predictive model.

### Results Comparison

| Metric | Logistic Regression | Clustering (ex-post) | Decision Tree |
|---|---|---|---|
| **Accuracy** | **0.882** | 0.874 | 0.854 |
| **Precision** | 0.862 | **0.895** | 0.828 |
| **Recall** | **0.876** | 0.815 | 0.853 |
| **Specificity** | **0.886** | 0.860 | 0.856 |
| **F1-Score** | **0.869** | 0.853 | 0.840 |

**Winner: Logistic Regression** — best overall balance across all metrics.

---

## 6. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── requirements.txt         # Dependencies (pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── Group_4_Heart_Disease_Prediction.ipynb
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data (train.csv, test.csv)
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

---

## 7. Execution Model

The full machine learning pipeline will eventually be executable through:

```bash
python src/main.py
```

The end goal is to expose a prediction endpoint via a web API so users can submit patient clinical features and receive a heart disease prediction.

---

## 8. Research Questions

1. Which clinical features are most associated with heart disease?
2. Can unsupervised learning identify meaningful patient risk profiles?
3. How does a supervised classification model compare to clustering-based approaches?

---

## 9. Key Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn (LogisticRegression, DecisionTreeClassifier, KMeans, PCA, StandardScaler)
- matplotlib
- seaborn

---

## 10. Next Steps

1. **Real-world data validation:** Apply models to real-world clinical datasets and compare performance against the synthetic data used here.
2. **Optimal threshold tuning:** Further refine the logistic regression decision threshold to minimize false positives in a clinical context.
3. **Alternative regularization:** Explore L1 (Lasso) and Elastic Net penalties to perform feature selection and potentially simplify the model.
4. **MLOps pipeline:** Modularize notebook code into `src/` modules, add data validation, model serialization, and deploy as a web-accessible API endpoint.
