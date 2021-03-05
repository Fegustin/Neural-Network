import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from kerastuner import RandomSearch, Hyperband, BayesianOptimization, Oracle


def build_model(hp):
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(units=hp.Range('units-input', mix_value=128, max_value=1024, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_tun():
    #   Получение данных
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #   Стандартизация входных данных
    x_train = x_train / 255
    x_test = x_test / 255

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    tun = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        directory='test_directory'
    )

    tun.search_space_summary()

    tun.search(
        x_train,
        y_train_cat,
        epochs=3,
        validation_split=0.2,
        verbose=1
    )
