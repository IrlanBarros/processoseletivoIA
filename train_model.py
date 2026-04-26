import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
input_shape = (28, 28, 1)
model_name = "model.h5"

# Carregar base de dados mnist e dividir entre conjunto de treino e de teste
print("\nCarregando base de dados mnist...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Converter valores de pixels para float e normalizar para valores entre 0 e 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Adicionar dimensão para o canal de cor exigido pelo TensorFlow/Keras
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("\nFormato de x_train:", x_train.shape)
print(x_train.shape[0], "amostras de treinamento")
print(x_test.shape[0], "amostras de teste")

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        
        # Primeira e segunda camada convolucional seguidas de Max Pooling 2 x 2
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),       
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten (transformar em vetor unidimensional), Dropout de 50% e camada densa de saída seguida de SoftMax
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

print("\nResumo do modelo:")
model.summary()

# Compilação e treinamento do modelo
batch_size = 128
epochs = 5

model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"])

print("\nTreinando modelo...")
model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_split=0.1)

# Avaliação de acurácia do modelo
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("\nTeste de perda:", loss)
print("Teste de acurácia:", accuracy)

# Salvar modelo no formato keras
model.save(model_name)