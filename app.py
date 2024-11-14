# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model('financial_advisor_model.h5')

# Iniciar la app Flask
app = Flask(__name__)

def classify_credit_score(score):
    """Clasifica el puntaje de crédito y genera un mensaje natural."""
    if score < 580:
        return "Tu puntaje de crédito es bajo. Es importante trabajar en mejorar tu historial de pagos y reducir tus deudas."
    elif 580 <= score < 670:
        return "Tu puntaje de crédito es promedio. Estás en el camino correcto, pero podrías mejorar tu estabilidad financiera reduciendo algunas deudas."
    elif 670 <= score < 740:
        return "Tu puntaje de crédito es bueno. Estás haciendo un gran trabajo manteniendo tus finanzas en orden."
    else:
        return "¡Excelente! Tu puntaje de crédito es muy alto. Estás en una excelente posición para acceder a los mejores productos financieros."


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Datos JSON recibidos
    features = np.array([data['features']])
    prediction = model.predict(features)
    
    # Convertir el resultado a tipo float para JSON
    credit_score = float(prediction[0][0])

       # Clasificación del puntaje
    message = classify_credit_score(credit_score)

    return jsonify({'predicted_credit_score': credit_score, 'message': message})

if __name__ == '__main__':
    app.run(port=5000)
