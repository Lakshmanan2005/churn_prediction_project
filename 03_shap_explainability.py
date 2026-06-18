"""
STEP 7: SHAP Explainability
This is what makes your project stand out - the ability to explain
WHY the model made a prediction, not just WHAT it predicted.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Load saved model and data ────────────────────────────────
model = joblib.load('churn_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

df = pd.read_csv('telco_churn_cleaned.csv')

# Recreate the same features used in training (must match exactly)
le_dict = joblib.load('label_encoders.pkl')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                             labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
df['HasMultipleServices'] = ((df['OnlineSecurity'] == 'Yes').astype(int) +
                              (df['OnlineBackup'] == 'Yes').astype(int) +
                              (df['DeviceProtection'] == 'Yes').astype(int) +
                              (df['TechSupport'] == 'Yes').astype(int))

for col, le in le_dict.items():
    df[col] = le.transform(df[col].astype(str))

X = df[feature_columns]

# Sample 500 rows for faster SHAP computation (common practice with large datasets)
X_sample = X.sample(500, random_state=42)

# ── Create SHAP explainer (TreeExplainer is fast for XGBoost) ─
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# ── GLOBAL EXPLANATION: Which features matter most overall? ──
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved SHAP summary plot (global feature importance)")

# ── Bar plot version (cleaner for resume/portfolio screenshots) ─
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved SHAP bar plot")

# ── LOCAL EXPLANATION: Why did the model predict churn for ONE customer? ─
# This is the killer feature for interviews - explain ONE prediction
sample_idx = 0
single_customer = X_sample.iloc[[sample_idx]]
prediction = model.predict_proba(single_customer)[0][1]
print(f"\nCustomer churn probability: {prediction:.2%}")

# Force plot explanation for this single customer
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_sample.iloc[sample_idx],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig('shap_single_customer.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved single-customer SHAP explanation")

# ── Print top features driving THIS prediction (text version) ──
shap_df = pd.DataFrame({
    'feature': X_sample.columns,
    'shap_value': shap_values[sample_idx]
}).sort_values('shap_value', key=abs, ascending=False)

print("\nTop 5 factors driving this customer's churn prediction:")
for _, row in shap_df.head(5).iterrows():
    direction = "increases" if row['shap_value'] > 0 else "decreases"
    print(f"  {row['feature']}: {direction} churn risk (SHAP={row['shap_value']:.3f})")
