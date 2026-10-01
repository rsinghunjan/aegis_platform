
#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) / 255.0
model = keras.Sequential([
    keras.layers.Input(shape=(28,28,1)),
    keras.layers.Conv2D(16,3,activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model')  # produces a TensorFlow SavedModel dir ready for TFServing / KServe
print("Saved model to saved_model/")
