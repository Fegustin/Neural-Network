import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras.layers import Dense


def price():
    #   Получение данных
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    #   Стандартизация входных данных
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = x_train - mean
    x_train = x_train / std
    x_test = x_test - mean
    x_test = x_test / std

    #   Создаем последовательную сеть и добавляем уровни сети
    model = keras.Sequential([
        Dense(400, input_shape=(x_train.shape[1],)),
        Dense(1)
    ])

    #   Информация о модели
    print(model.summary())

    #   Компилируем модель
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    #   Обучаем сеть
    model.fit(x_train, y_train, batch_size=12, epochs=50, validation_split=0.2, verbose=1)

    model.evaluate(x_test, y_test, verbose=1)

    pred = model.predict(x_test)

    print(pred[1][0], y_test[1])

    print(pred[50][0], y_test[50])

    print(pred[100][0], y_test[100])
