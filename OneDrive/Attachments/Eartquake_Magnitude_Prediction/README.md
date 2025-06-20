# Eartquake_Magnitude_Prediction
# 🌍 Earthquake Magnitude Prediction App

A complete end-to-end machine learning web application that predicts the magnitude of an earthquake based on seismic and geographical inputs. Built with **Flask**, **CatBoost**, and **scikit-learn**, and served through a simple, elegant web interface.This repository contains a Flask web application that uses a CatBoost regression model to predict earthquake magnitudes. The model is trained using a dataset containing various features related to earthquakes. This project demonstrates the process of loading data, preprocessing it, training a machine learning model, and deploying the model as a web service using Flask.


---

## 📌 Features

- ✅ Predicts earthquake magnitude (`xm`) using 11 user-provided parameters  
- ✅ Web interface built with HTML + CSS and Flask templating  
- ✅ CatBoostRegressor with k-fold cross-validation  
- ✅ Model serialization using `pickle`  
- ✅ Clean UI and real-time predictions  

---

## 🧠 Technologies Used

- **Flask 2.3.2**
- **scikit-learn**
- **CatBoost**
- **NumPy, Pandas**
- **HTML + CSS**
- **Pickle** 
- **Data Preprocessing:** Load and clean the dataset, handle missing values, and prepare categorical features.
- **Model Training:** Train a CatBoost regression model using k-fold cross-validation for robust performance evaluation.
- **Model Evaluation:** Evaluate the model using metrics such as Mean Squared Error (MSE), R-squared (R²), and Mean Absolute Error (MAE).
- **Web Interface:** Provide a user-friendly web interface to input features and get earthquake magnitude predictions.
- **Model Persistence:** Save and load the trained model using pickle for easy reuse.

## File Descriptions

- `app.py`: The main Flask application file containing routes and model handling code.
- `dataset.csv`: The dataset file (not included, needs to be added by the user).
- `catboost_model.pkl`: The serialized CatBoost model file (generated after training).
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: The list of Python dependencies required to run the application.


---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/Ed-Gigaversity/Earthquake_magnitude_prediction.git
cd Eartquake_Magnitude_Prediction

## 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS/Linux

## 3. Install dependencies
pip install -r requirements.txt

### 4. Run the application
python app.py
Visit the web app at: http://127.0.0.1:5000/