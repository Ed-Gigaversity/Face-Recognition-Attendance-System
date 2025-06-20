
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import pickle
import time

app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    df = pd.read_csv('dataset.csv')

    # Drop unnecessary columns
    df = df.drop(['id', 'mw', 'dist', 'date', 'time'], axis=1)

    # Handle missing values
    df.fillna('missing', inplace=True)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Separate features and target
    y = df['xm']
    X = df.drop('xm', axis=1)

    return X, y, categorical_cols

# Train the model
def train_model(X, y, categorical_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=categorical_cols,
        silent=True
    )

    print("Training with cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

    print("CV MSE Mean:", -cv_scores.mean())
    print("CV R² Mean:", cv_r2_scores.mean())

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    print(f"Test R²: {r2_score(y_test, y_pred):.3f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"Training Time: {end - start:.2f} seconds")

    return model, X.columns.tolist()

# Load and train model
X, y, categorical_cols = load_and_preprocess_data()
model, feature_order = train_model(X, y, categorical_cols)

# Save model
with open('catboost_model.pkl', 'wb') as f:
    pickle.dump((model, feature_order), f)

# Load model
with open('catboost_model.pkl', 'rb') as f:
    model, feature_order = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        input_dict = {
            'lat': float(data.get('lat', 0)),
            'long': float(data.get('long', 0)),
            'country': data.get('country', 'missing'),
            'city': data.get('city', 'missing'),
            'area': data.get('area', 'missing'),
            'direction': data.get('direction', 'missing'),
            'depth': float(data.get('depth', 0)),
            'md': float(data.get('md', 0)),
            'richter': float(data.get('richter', 0)),  # ✅ fixed spelling
            'ms': float(data.get('ms', 0)),
            'mb': float(data.get('mb', 0))
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_order]  # ensure correct column order
        input_df.fillna('missing', inplace=True)

        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Predicted Magnitude: {prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
