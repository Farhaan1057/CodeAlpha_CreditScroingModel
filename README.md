# 💳 Credit Scoring Model — CodeAlpha ML Internship

> **Task 1** of the CodeAlpha Machine Learning Internship Program

## 📌 Objective
Predict an individual's **creditworthiness** (good/bad credit risk) using past financial data, applying multiple classification algorithms and evaluating them with industry-standard metrics.

---

## 🗂 Project Structure
```
CodeAlpha_CreditScoringModel/
├── credit_scoring_model.py      # Main Python script (run this)
├── credit_scoring_model.ipynb   # Jupyter Notebook (visual walkthrough)
├── requirements.txt             # Dependencies
├── README.md
└── outputs/                     # Generated after running
    ├── credit_data.csv
    ├── eda_distributions.png
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

---

## 🔧 Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CreditScoringModel
cd CodeAlpha_CreditScoringModel

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate      


# 4. Run the script
python credit_scoring_model.py

# OR open the notebook
jupyter notebook credit_scoring_model.ipynb
```

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `age` | Applicant age |
| `income` | Annual income |
| `employment_years` | Years at current job |
| `credit_util_ratio` | Credit utilization (0–1) |
| `num_late_payments` | Count of late payments |
| `debt_to_income` | Debt-to-income ratio |
| `savings_balance` | Current savings |
| `loan_amount` | Requested loan amount |
| `existing_loans` | Number of active loans |
| + 8 engineered features | Derived ratios & flags |

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable rule-based model |
| **Random Forest** | Ensemble of decision trees |
| Gradient Boosting | Boosted ensemble (often best) |

---

## 📈 Evaluation Metrics
- **Accuracy** — overall correct predictions
- **Precision** — of predicted good credit, how many actually are
- **Recall** — of actual good credit, how many were caught
- **F1-Score** — harmonic mean of precision & recall
- **ROC-AUC** — area under the ROC curve (discrimination ability)
- **5-Fold Cross-Validation** — generalization check

---

## 🔍 Key Findings
- Credit utilization ratio and debt-to-income are the **strongest predictors**
- Late payment history is a major **red flag**
- Random Forest and Gradient Boosting consistently achieve **ROC-AUC > 0.90**
- Feature engineering (ratios and interaction terms) improves performance

---

## Author 
Muhammad Farhan
CodeAlpha ML Internship — 2026