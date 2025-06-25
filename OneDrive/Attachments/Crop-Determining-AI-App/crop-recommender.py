# ✅ Final Debugged & Optimized crop-recommender.py

import numpy as np
import pandas as pd
from joblib import dump
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# 1. Load dataset
print("📥 Loading dataset...")
crop_df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv")

# 2. Handle class imbalance: Upsample maize class
print("🔁 Upsampling 'maize' class...")
maize_df = crop_df[crop_df['crop'] == 'maize']
other_df = crop_df[crop_df['crop'] != 'maize']
maize_upsampled = resample(maize_df, replace=True, n_samples=500, random_state=42)
balanced_df = pd.concat([other_df, maize_upsampled])

# 3. Encode labels
print("🔤 Encoding labels...")
le = LabelEncoder()
balanced_df['crop'] = le.fit_transform(balanced_df['crop'])
X = balanced_df.drop("crop", axis=1)
y = balanced_df["crop"]

# 4. Train-test split
print("📊 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Random Forest - Hyperparameter Tuning
print("🌲 Tuning Random Forest model...")
random_grid_rf = {
    'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid_rf,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

# 6. XGBoost - Hyperparameter Tuning
print("⚡ Tuning XGBoost model...")
random_grid_xgb = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.2, 0.3],
    "colsample_bytree": [0.3, 0.5, 0.7]
}
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=random_grid_xgb,
                                n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
xgb_random.fit(X_train, y_train)
best_xgb = xgb_random.best_estimator_

# 7. Evaluate models
acc_rf = accuracy_score(y_test, best_rf.predict(X_test))
acc_xgb = accuracy_score(y_test, best_xgb.predict(X_test))
print(f"\n✅ Random Forest Accuracy: {acc_rf:.4f}")
print(f"✅ XGBoost Accuracy: {acc_xgb:.4f}")

# 8. Save best model
best_model = best_rf if acc_rf > acc_xgb else best_xgb
model_name = "Crop-Recommender.pkl"
encoder_name = "label_encoder.pkl"
dump(best_model, model_name)
dump(le, encoder_name)

print("\n🎉 Model and label encoder saved successfully!")
print(f"📁 Saved: {model_name}, {encoder_name}")
