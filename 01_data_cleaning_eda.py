"""
STEP 1 & 2: Data Loading, Cleaning, and EDA
Customer Churn Prediction Project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ────────────────────────────────────────────────
df = pd.read_csv('telco_churn_cleaned.csv')
print("Shape:", df.shape)
print("\nFirst look:\n", df.head())

# ── Basic Info ───────────────────────────────────────────────
print("\nInfo:")
df.info()

print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ── Clean: TotalCharges has blank strings, not NaN ──────────
# This is a classic real-world data issue you should mention in interviews
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)
print("\nMissing after conversion:", df['TotalCharges'].isnull().sum())

# These 11 missing rows are customers with tenure = 0 (brand new customers)
print(df[df['TotalCharges'].isnull()][['tenure', 'MonthlyCharges', 'TotalCharges']])

# Fill missing TotalCharges with 0 (since tenure=0, they haven't been charged yet)
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# ── Drop customerID (not useful for modeling) ──────────────
#df.drop('customerID', axis=1, inplace=True)

# ── Target variable distribution (THIS IS YOUR IMBALANCE) ──
print("\nChurn distribution:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)
# Output: No ~73%, Yes ~27% -> This IS the imbalance problem you'll solve with SMOTE

# ── EDA Visualizations ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Churn distribution
sns.countplot(data=df, x='Churn', ax=axes[0,0], palette='Set2')
axes[0,0].set_title('Churn Distribution (Imbalanced)')

# 2. Churn by Contract type
sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[0,1], palette='Set2')
axes[0,1].set_title('Churn by Contract Type')
axes[0,1].tick_params(axis='x', rotation=20)

# 3. Tenure distribution by churn
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, ax=axes[0,2], palette='Set2')
axes[0,2].set_title('Tenure Distribution by Churn')

# 4. Monthly charges by churn
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0], palette='Set2')
axes[1,0].set_title('Monthly Charges by Churn')

# 5. Internet Service vs Churn
sns.countplot(data=df, x='InternetService', hue='Churn', ax=axes[1,1], palette='Set2')
axes[1,1].set_title('Churn by Internet Service')

# 6. Payment method vs churn
sns.countplot(data=df, y='PaymentMethod', hue='Churn', ax=axes[1,2], palette='Set2')
axes[1,2].set_title('Churn by Payment Method')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=100)
print("\nSaved EDA charts to eda_overview.png")

# Save cleaned data for next step
df.to_csv('telco_churn_cleaned.csv', index=False)
print("\nCleaned data saved!")
