"""
STEP 3, 4, 5, 6: Feature Engineering, SMOTE, XGBoost Training, Evaluation
Customer Churn Prediction Project
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import joblib

# ── Load cleaned data ────────────────────────────────────────
df = pd.read_csv('telco_churn_cleaned.csv')

# ── STEP 3: Feature Engineering ─────────────────────────────

# 3a. Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3b. Create new engineered features (THIS shows skill in interviews)
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 avoids divide by zero
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                             labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
df['HasMultipleServices'] = ((df['OnlineSecurity'] == 'Yes').astype(int) +
                              (df['OnlineBackup'] == 'Yes').astype(int) +
                              (df['DeviceProtection'] == 'Yes').astype(int) +
                              (df['TechSupport'] == 'Yes').astype(int))

# 3c. Encode categorical variables
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols += ['TenureGroup']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le  # save encoders for later use in deployment

print("Final feature columns:", df.columns.tolist())
print("\nShape after feature engineering:", df.shape)

# ── Split features & target ─────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")

# ── STEP 4: Handle Class Imbalance with SMOTE ───────────────
# IMPORTANT: Apply SMOTE only to training data, never to test data!
print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {y_train_sm.value_counts().to_dict()}")
# Now both classes are balanced 50/50 in training data

# ── STEP 5: Train XGBoost Model ──────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42
)

model.fit(X_train_sm, y_train_sm)

# ── STEP 6: Evaluate Model ───────────────────────────────────
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE ON TEST SET (unseen data)")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── Plot ROC Curve ───────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.3f})', color='#2196F3', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Customer Churn Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=100)
print("\nROC curve saved!")

# ── Feature Importance (basic, before SHAP) ─────────────────
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(importance['feature'][:10][::-1], importance['importance'][:10][::-1], color='#26A69A')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (XGBoost)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
print("Feature importance chart saved!")

# ── Save model and encoders for deployment ──────────────────
joblib.dump(model, 'churn_model.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("\nModel and encoders saved for deployment!")
