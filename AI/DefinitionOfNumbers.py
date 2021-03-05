import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout


def definition():
    #   Получение данных
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #   Стандартизация входных данных
    x_train = x_train / 255
    x_test = x_test / 255

    #   Преобразуем метки в категории
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    #   Отображение первых 25 изображений из будущей выборки
    # plt.figure(figsize=(10, 5))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(x_train[i], cmap=plt.cm.binary)
    # plt.show()

    #   Создаем последовательную сеть и добавляем уровни сети
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(1000, activation='relu'),
        Dropout(0.8),
        Dense(10, activation='softmax')
    ])

    #   Информация о модели
    print(model.summary())

    my_adam = keras.optimizers.Adam(learning_rate=0.1)
    my_sgd = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

    #   Компилируем модель
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    from sklearn.model_selection import train_test_split

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, train_size=0.2)

    #   Обучаем сеть
    his = model.fit(x_train_split, y_train_split, batch_size=32, epochs=10,
                    validation_data=(x_val_split, y_val_split))

    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.show()

    #   Оцениваем качество обучения сети на тестовых данных
    model.evaluate(x_test, y_test_cat)
