import tensorflow as tf
import os

# 1. Carregar o modelo já treinado
model_name = "model.h5"
model = tf.keras.models.load_model(model_name)

# 2. Criar o conversor TFLite a partir do modelo Keras
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. ATIVAR a Otimização de Faixa Dinâmica (Dynamic Range Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Converter o modelo
tflite_model_quant = converter.convert()

# 5. Salvar o modelo quantizado
tflite_model_name = "model.tflite"
with open(tflite_model_name, "wb") as f:
    f.write(tflite_model_quant)

print(f"Modelo otimizado salvo como: {tflite_model_name}")

# Comparação entre o modelo original e o otimizado
original_size = os.path.getsize(model_name) / (1024 * 1024)  # MB
optimized_size = os.path.getsize(tflite_model_name) / (1024 * 1024)  # MB
reduction = (1 - (optimized_size / original_size)) * 100

print(f"\nResumo da Otimização:")
print(f"Tamanho original (model.h5): {original_size:.2f} MB")
print(f"Tamanho otimizado (model.tflite): {optimized_size:.2f} MB")
print(f"Redução de tamanho: {reduction:.2f}%")