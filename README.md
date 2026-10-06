# Credit Card Fraud Detection

An end-to-end machine learning project focused on detecting potentially fraudulent credit card transactions using Python and scikit-learn, with an emphasis on reliable evaluation, explainability, and production-oriented ML engineering.

> **Project status:** In development.

## Overview

Credit card fraud detection is an imbalanced classification problem where fraudulent transactions represent only a small fraction of total transactions. A useful detection system must identify fraudulent activity while managing false positives that can unnecessarily flag legitimate customers.

This project explores the development of a reproducible machine learning pipeline for transaction classification, from data exploration and preprocessing to model evaluation, explainability, and API-based inference.

The goal is to build a maintainable ML system rather than a standalone model-training notebook.

## Objectives

- Develop a structured and reproducible machine learning workflow.
- Handle class imbalance in fraud detection.
- Prevent data leakage through appropriate preprocessing and data splitting.
- Evaluate models using metrics suitable for imbalanced classification.
- Analyze false positives and false negatives.
- Investigate feature contributions using model explainability techniques.
- Expose trained model predictions through a REST API.
- Improve reliability through automated testing and deployment-oriented practices.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Model explainability | SHAP |
| API development | FastAPI, Pydantic |
| Model persistence | Joblib |
| Testing | Pytest |
| Development environment | Jupyter Notebook, VS Code |
| Deployment tooling | Docker (planned) |

## Dataset

This project uses the publicly available Credit Card Fraud Detection dataset.

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymized credit card transactions made by European cardholders in September 2013.

Key characteristics:

- 284,807 transactions.
- 492 fraudulent transactions.
- 30 input features: `Time`, `Amount`, and `V1` through `V28`.
- Target column: `Class`, where `0` represents a legitimate transaction and `1` represents a fraudulent transaction.

The dataset is highly imbalanced, making accuracy alone an insufficient measure of model quality.

Download the dataset from Kaggle and place `creditcard.csv` in the `data/` directory.

## Project Architecture

```text
Historical Transaction Data
            |
            v
     Exploratory Analysis
            |
            v
  Data Preprocessing Pipeline
            |
            v
    Model Training
            |
            v
    Model Evaluation
            |
            v
  Explainability & Error Analysis
            |
            v
     Saved Model Artifact
            |
            v
      FastAPI REST API
            |
            v
  Transaction Prediction Response
```

Training and inference are separate stages. The training workflow fits the model and saves the resulting artifact. The API loads the saved artifact and uses it to generate predictions for incoming transaction features without retraining for every request.

## Repository Structure

```text
ml_fraud_detection/
├── app/
│   └── main.py
├── data/
│   └── creditcard.csv
├── docker/
├── models/
├── notebooks/
│   └── eda.ipynb
├── reports/
│   └── error_analysis.md
├── src/
│   ├── pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── tests/
│   ├── test_api.py
│   └── test_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

- `notebooks/`: Exploratory data analysis and visualization.
- `src/pipeline.py`: Reusable preprocessing and model pipeline components.
- `src/train.py`: Model training and artifact generation.
- `src/evaluate.py`: Evaluation on held-out data.
- `src/explain.py`: Model interpretation and feature contribution analysis.
- `app/main.py`: API application for transaction inference.
- `tests/`: Automated tests for pipeline behavior and API functionality.
- `reports/`: Documented findings, model errors, and analysis.
- `data/`: Local dataset storage.
- `models/`: Saved model artifacts.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version.
- Git.
- A Kaggle account to access the dataset.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-fraud-detection.git
cd ml-fraud-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it here:

```text
data/creditcard.csv
```

### 5. Train the model

Once the training script and its dependencies are configured:

```bash
python src/train.py
```

The training workflow should fit the configured pipeline, evaluate the model, and save the model artifacts for subsequent inference.

### 6. Run the API

After a compatible trained model artifact has been generated and the API is configured to load it:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to submit a JSON request containing the required transaction features and inspect the prediction response.

> These commands describe the intended workflow. They should be verified against the current implementation before the project is presented as runnable end to end.

## Evaluation Strategy

Because fraudulent transactions are rare, the project prioritizes metrics that reveal how well the model identifies the minority class.

- **Precision:** The proportion of predicted fraud cases that are actually fraudulent.
- **Recall:** The proportion of actual fraud cases successfully detected.
- **F1-score:** The harmonic mean of precision and recall.
- **Average precision:** A summary measure of performance across precision-recall thresholds.
- **Confusion matrix:** A breakdown of true positives, true negatives, false positives, and false negatives.

The decision threshold should be selected using validation data and an explicit understanding of the costs of missed fraud and incorrectly flagged legitimate transactions.

Final results should be reported only after running the evaluation workflow on held-out data.

## Explainability and Error Analysis

A fraud detection model should be evaluated beyond its aggregate metrics.

The planned analysis includes:

- Investigating false positives and false negatives.
- Identifying patterns in misclassified transactions.
- Examining feature contributions using SHAP where appropriate.
- Comparing model behavior under different classification thresholds.
- Documenting limitations and potential sources of bias or instability.

The objective is to understand where the model performs well, where it fails, and how those failures could affect a real transaction-monitoring workflow.

## Testing and Reliability

Automated tests are included in the project structure to support validation of the preprocessing pipeline and prediction API.

The intended checks include:

- Consistent feature handling between training and inference.
- Input validation and meaningful error responses.
- Correct prediction output structure.
- Reproducible preprocessing behavior.
- Basic API and pipeline regression tests.

Test coverage and results will be documented after the implementation has been executed and verified.

## Limitations

- The dataset contains anonymized features, so predictions do not directly explain transactions using original merchant or cardholder attributes.
- The data represents a historical snapshot and may not reflect current fraud patterns.
- Class imbalance makes the choice of evaluation metrics and classification threshold particularly important.
- A model's output is a statistical prediction, not proof that a transaction is fraudulent.
- The project is an educational ML engineering implementation, not a production financial fraud prevention service.

## Future Improvements

- Compare logistic regression with tree-based models.
- Add cross-validation and systematic hyperparameter tuning.
- Tune the decision threshold using validation data and explicit cost assumptions.
- Evaluate probability calibration.
- Add more comprehensive model and API tests.
- Containerize the service using Docker.
- Introduce automated CI checks.
- Document model versioning, reproducibility, and inference monitoring.

## Author

**Adrin Joseph**

Machine Learning | Python | ML Engineering
---

*This project is being developed as part of a hands-on exploration of end-to-end machine learning engineering, from data preparation and evaluation to model serving and reliability.*
