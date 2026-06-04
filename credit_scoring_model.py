# ============================================================
# CodeAlpha Internship — Task 1: Credit Scoring Model
# Author: [Your Name]
# Description: Predict creditworthiness using classification algorithms
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE REALISTIC DATASET (~30% bad credit)
# ─────────────────────────────────────────────
def generate_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    n = n_samples

    age               = np.random.randint(18, 75, n)
    income            = np.random.normal(55000, 25000, n).clip(10000, 200000)
    employment_years  = np.random.randint(0, 35, n)
    num_credit_lines  = np.random.randint(1, 15, n)
    credit_util_ratio = np.random.beta(2, 5, n)
    num_late_payments = np.random.poisson(1.2, n)
    debt_to_income    = np.random.beta(3, 5, n)
    loan_amount       = np.random.normal(15000, 8000, n).clip(1000, 60000)
    loan_tenure_yrs   = np.random.randint(1, 10, n)
    existing_loans    = np.random.randint(0, 5, n)
    savings_balance   = np.random.exponential(8000, n).clip(0, 100000)
    has_mortgage      = np.random.binomial(1, 0.4, n)
    has_car_loan      = np.random.binomial(1, 0.3, n)

    # Weighted credit score formula
    score = (
        0.30 * (1 - credit_util_ratio)
      + 0.25 * (1 - debt_to_income)
      + 0.20 * (1 - np.minimum(num_late_payments, 10) / 10)
      + 0.10 * np.log1p(income) / np.log1p(200000)
      + 0.10 * np.log1p(savings_balance) / np.log1p(100000)
      + 0.05 * employment_years / 35
    )
    noise = np.random.normal(0, 0.08, n)
    # threshold=0.68 gives realistic ~30% bad credit rate
    creditworthy = ((score + noise) >= 0.68).astype(int)

    df = pd.DataFrame({
        'age':               age,
        'income':            income.round(2),
        'employment_years':  employment_years,
        'num_credit_lines':  num_credit_lines,
        'credit_util_ratio': credit_util_ratio.round(4),
        'num_late_payments': num_late_payments,
        'debt_to_income':    debt_to_income.round(4),
        'loan_amount':       loan_amount.round(2),
        'loan_tenure_yrs':   loan_tenure_yrs,
        'existing_loans':    existing_loans,
        'savings_balance':   savings_balance.round(2),
        'has_mortgage':      has_mortgage,
        'has_car_loan':      has_car_loan,
        'creditworthy':      creditworthy
    })

    # Realistic missing values (~2% per column)
    for col in ['income', 'employment_years', 'savings_balance']:
        mask = np.random.rand(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def run_eda(df):
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\nDataset shape  : {df.shape}")
    print(f"Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    vc = df['creditworthy'].value_counts()
    print(f"\nClass balance  : {vc[0]} bad credit ({vc[0]/len(df):.1%})  |  {vc[1]} good credit ({vc[1]/len(df):.1%})")
    print("\nDescriptive stats:")
    print(df.describe().round(2))

    # Distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Feature Distributions by Credit Class", fontsize=15, fontweight='bold')
    features = ['age','income','employment_years','credit_util_ratio',
                'num_late_payments','debt_to_income','loan_amount',
                'savings_balance','num_credit_lines']
    for ax, feat in zip(axes.flat, features):
        for label, color in zip([0,1], ['#e74c3c','#2ecc71']):
            ax.hist(df[df.creditworthy==label][feat].dropna(), bins=30, alpha=0.6,
                    color=color, label=('Bad Credit' if label==0 else 'Good Credit'))
        ax.set_title(feat.replace('_',' ').title()); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[Saved] eda_distributions.png")

    # Correlation heatmap
    plt.figure(figsize=(12, 9))
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, square=True, cbar_kws={'shrink':0.7})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Saved] correlation_matrix.png")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()
    df['monthly_income']         = df['income'] / 12
    df['monthly_loan_payment']   = df['loan_amount'] / (df['loan_tenure_yrs'] * 12)
    df['payment_income_ratio']   = df['monthly_loan_payment'] / df['monthly_income'].replace(0, np.nan)
    df['savings_to_income']      = df['savings_balance'] / df['income'].replace(0, np.nan)
    df['late_payment_flag']      = (df['num_late_payments'] > 2).astype(int)
    df['high_util_flag']         = (df['credit_util_ratio'] > 0.7).astype(int)
    df['total_loan_burden']      = df['existing_loans'] + df['has_mortgage'] + df['has_car_loan']
    df['age_income_interaction'] = df['age'] * df['income'] / 1e6
    return df


# ─────────────────────────────────────────────
# 4. PREPARE DATA
# ─────────────────────────────────────────────
def prepare_data(df):
    X = df.drop(columns=['creditworthy'])
    y = df['creditworthy']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain : {X_train.shape[0]} samples  |  Test : {X_test.shape[0]} samples")
    print(f"Bad credit in train : {(1-y_train.mean()):.1%}  |  in test : {(1-y_test.mean()):.1%}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 5. BUILD & EVALUATE
# ─────────────────────────────────────────────
def build_pipeline(classifier):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   classifier)
    ])

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, results):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(pipeline, X_train, y_train,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring='roc_auc')

    print(f"\n{'─'*50}\n  {name}\n{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  CV AUC    : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Bad Credit','Good Credit'], zero_division=0)}")

    results[name] = {
        'pipeline': pipeline, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc, 'cv_auc_mean': cv.mean()
    }
    return results


# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────
def plot_results(results, X_test, y_test):
    colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']

    # ROC curves + metric bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for (name, r), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
    axes[0].plot([0,1],[0,1],'k--',lw=1)
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves — All Models', fontweight='bold')
    axes[0].legend(loc='lower right'); axes[0].grid(alpha=0.3)

    metrics = ['accuracy','precision','recall','f1','roc_auc']
    x = np.arange(len(metrics)); w = 0.18
    for i, (name, r) in enumerate(results.items()):
        axes[1].bar(x + i*w, [r[m] for m in metrics], w, label=name, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x + w*1.5)
    axes[1].set_xticklabels([m.replace('_','\n') for m in metrics])
    axes[1].set_ylim(0, 1.05); axes[1].set_title('Model Comparison', fontweight='bold')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3); axes[1].set_ylabel('Score')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Saved] model_comparison.png")

    # Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Saved] confusion_matrices.png")

    # Feature importance (Random Forest)
    if 'Random Forest' in results:
        rf = results['Random Forest']['pipeline'].named_steps['model']
        fi_df = pd.DataFrame({'feature': X_test.columns, 'importance': rf.feature_importances_})
        fi_df = fi_df.sort_values('importance', ascending=True).tail(15)
        plt.figure(figsize=(10, 7))
        bar_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
        plt.barh(fi_df['feature'], fi_df['importance'], color=bar_colors)
        plt.title('Random Forest — Top Feature Importances', fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[Saved] feature_importance.png")


# ─────────────────────────────────────────────
# 7. DEMO PREDICTION
# ─────────────────────────────────────────────
def predict_applicant(best_pipeline, feature_names):
    print("\n" + "="*60)
    print("  DEMO: Predicting a New Loan Applicant")
    print("="*60)
    applicant = {
        'age': 34, 'income': 62000, 'employment_years': 7,
        'num_credit_lines': 4, 'credit_util_ratio': 0.35,
        'num_late_payments': 1, 'debt_to_income': 0.28,
        'loan_amount': 18000, 'loan_tenure_yrs': 5,
        'existing_loans': 1, 'savings_balance': 12000,
        'has_mortgage': 0, 'has_car_loan': 1,
        'monthly_income': 62000/12,
        'monthly_loan_payment': 18000/(5*12),
        'payment_income_ratio': (18000/(5*12))/(62000/12),
        'savings_to_income': 12000/62000,
        'late_payment_flag': 0, 'high_util_flag': 0,
        'total_loan_burden': 2,
        'age_income_interaction': 34*62000/1e6,
    }
    app_df = pd.DataFrame([applicant])[feature_names]
    prob   = best_pipeline.predict_proba(app_df)[0][1]
    pred   = best_pipeline.predict(app_df)[0]
    print(f"\n  Applicant profile:")
    for k in ['age','income','employment_years','credit_util_ratio','num_late_payments','debt_to_income']:
        print(f"    {k:28s}: {applicant[k]}")
    print(f"\n  Credit probability : {prob:.2%}")
    print(f"  Decision           : {'✅ APPROVED (Good Credit)' if pred==1 else '❌ DECLINED (Bad Credit)'}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  CodeAlpha — Credit Scoring Model")
    print("="*60)

    print("\n[1/6] Generating dataset …")
    df_raw = generate_dataset(5000)
    df_raw.to_csv('credit_data.csv', index=False)
    print(f"      Saved → credit_data.csv  ({len(df_raw)} rows)")

    print("\n[2/6] Running EDA …")
    run_eda(df_raw)

    print("\n[3/6] Feature engineering …")
    df = feature_engineering(df_raw)

    print("\n[4/6] Train/test split …")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("\n[5/6] Training & evaluating models …")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    results = {}
    for name, clf in models.items():
        results = evaluate_model(name, build_pipeline(clf), X_train, X_test, y_train, y_test, results)

    print("\n[6/6] Generating visualisations …")
    plot_results(results, X_test, y_test)

    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\n{'='*60}")
    print(f"  🏆 Best Model : {best_name}")
    print(f"     ROC-AUC   : {results[best_name]['roc_auc']:.4f}")
    print(f"     F1-Score  : {results[best_name]['f1']:.4f}")
    print(f"{'='*60}")

    predict_applicant(results[best_name]['pipeline'], X_train.columns.tolist())
    print("\n✅ All done! Charts saved as PNG files.")

if __name__ == '__main__':
    main()