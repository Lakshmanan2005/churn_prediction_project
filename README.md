# 📊 Customer Churn Prediction with XGBoost + SHAP

End-to-end machine learning project that predicts customer churn for a telecom company and explains *why* each prediction was made using SHAP, deployed as an interactive Streamlit web app.

## 🎯 Problem Statement
Customer churn (customers leaving) directly hurts revenue. This project predicts which customers are at risk of churning **and** explains the key drivers behind each prediction — critical for actionable, trustworthy business decisions in BFSI, telecom, and SaaS industries.

## 📁 Dataset
[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features (demographics, account info, services subscribed).

## 🔧 Tech Stack
- **Data Processing:** Pandas, NumPy
- **Imbalance Handling:** SMOTE (imbalanced-learn)
- **Modeling:** XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## 📈 Results
| Metric | Score |
|---|---|
| AUC-ROC | 0.834 |
| Accuracy | 76.9% |
| Precision | 0.56 |
| Recall | 0.62 |
| F1-Score | 0.59 |

## 🔍 Key Insights (from SHAP)
- **Contract type** is the #1 churn driver — month-to-month customers churn far more than 1-2 year contract holders.
- **Tenure** strongly predicts loyalty — newer customers are much higher risk.
- **Fiber optic internet** customers churn more than DSL customers (likely due to pricing/service issues).
- Customers **without tech support / online security** add-ons churn more — these act as "stickiness" features.

## 🚀 Project Pipeline
1. **Data Cleaning** — fixed blank `TotalCharges` strings, handled 11 missing values for zero-tenure customers
2. **EDA** — visualized churn patterns across contract type, tenure, charges, services
3. **Feature Engineering** — created `AvgMonthlySpend`, `TenureGroup`, `HasMultipleServices`
4. **Imbalance Handling** — applied SMOTE only on training data (73%/27% → 50%/50%)
5. **Model Training** — XGBoost classifier with tuned hyperparameters
6. **Evaluation** — AUC-ROC, precision/recall/F1, confusion matrix, ROC curve
7. **Explainability** — SHAP global (summary plot) and local (force plot) explanations
8. **Deployment** — interactive Streamlit app for real-time predictions with live SHAP explanations

## 💻 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python 01_data_cleaning_eda.py
python 02_feature_eng_train_evaluate.py
python 03_shap_explainability.py

# Launch the web app
streamlit run app.py
```

## 📂 Project Structure
```
├── 01_data_cleaning_eda.py          # Data cleaning + exploratory analysis
├── 02_feature_eng_train_evaluate.py # Feature engineering + XGBoost training
├── 03_shap_explainability.py        # SHAP global & local explanations
├── app.py                            # Streamlit deployment app
├── requirements.txt
├── churn_model.pkl                   # Saved trained model
├── label_encoders.pkl                # Saved encoders for deployment
└── feature_columns.pkl               # Feature column order
```

## 🌐 Live Demo
[Add your Streamlit Cloud / Hugging Face Spaces link here after deployment]

## 👤 Author
Your Name — [LinkedIn](https://linkedin.com/in/lakshmanan-c-22342b292) | [Portfolio](https://lakshmanan2005.github.io/)
