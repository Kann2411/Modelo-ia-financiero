# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# Datos de entrada (características)
features = np.array([
    [1000, 300, 100, 200, 150, 50, 300, 23],  # 18-25 años, gastos bajos
    [1200, 400, 150, 250, 150, 100, 350, 27], # 25-40 años, gastos moderados
    [1500, 500, 200, 300, 180, 150, 400, 35], # 25-40 años, gastos medios
    [2000, 600, 300, 350, 200, 200, 500, 45], # 40-60 años, gastos altos
    [1800, 400, 250, 250, 150, 100, 400, 50], # 40-60 años, gastos bajos-moderados
    [500, 300, 100, 100, 100, 50, 200, 65],   # 60+ años, gastos bajos
])

# Etiquetas (puntaje de crédito)
labels = np.array([
    550,  # Edad 18-25, crédito bajo
    690,  # Edad 25-40, crédito medio
    720,  # Edad 25-40, crédito medio-alto
    750,  # Edad 40-60, crédito alto
    700,  # Edad 40-60, crédito medio-alto
    760   # Edad 60+, crédito muy alto
])

# Definir el modelo
model = Sequential()
model.add(Dense(64, input_shape=(8,), activation='relu'))  # Capa de entrada
model.add(Dense(32, activation='relu'))                    # Capa oculta
model.add(Dense(1))                                        # Capa de salida

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(features, labels, epochs=100, batch_size=1)

model.save('financial_advisor_model.h5')
