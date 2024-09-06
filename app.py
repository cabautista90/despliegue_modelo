from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('mejor_modelo.pkl')  # Asegúrate de que 'mejor_modelo.pkl' esté en el mismo directorio

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos de la solicitud POST
    data = request.get_json(force=True)
    # Realizar la predicción usando el modelo cargado
    prediction = model.predict([data['input']])
    # Devolver la predicción en formato JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
