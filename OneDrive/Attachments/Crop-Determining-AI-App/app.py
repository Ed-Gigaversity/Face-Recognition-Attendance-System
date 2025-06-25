import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("Crop-Recommender.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['nitrogenBox']),
            float(request.form['phosphorusBox']),
            float(request.form['potassiumBox']),
            float(request.form['temperatureBox']),
            float(request.form['humidityBox']),
            float(request.form['phBox']),
            float(request.form['rainfallBox'])
        ]
        
        prediction = model.predict([np.array(features)])
        crop_name = label_encoder.inverse_transform(prediction)[0]

        # Get top 3 predictions
        probs = model.predict_proba([np.array(features)])
        top3 = np.argsort(probs[0])[-3:][::-1]
        top_predictions = [(label_encoder.inverse_transform([i])[0], probs[0][i]) for i in top3]

        top_pred_text = "<br>".join([f"{name}: {score*100:.2f}%" for name, score in top_predictions])
        result = f"The recommended crop is: 🌱 <b>{crop_name.upper()}</b><br><br><b>Top 3 Predictions:</b><br>{top_pred_text}"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
