from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.metrics import accuracy_score

def one_hot_encode(num_classes, input_values):
    encoded_values = np.zeros((len(input_values), num_classes))
    for i, value in enumerate(input_values):
        encoded_values[i, value] = 1
    return encoded_values

(x_train, y_train_temp), (x_test, y_test_temp) = cifar10.load_data()

y_train = one_hot_encode(10, y_train_temp)
y_test  = one_hot_encode(10, y_test_temp)

model = Sequential()

model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=30)

y_pred = model.predict(x_test)

print(f"Accuracy: {100*accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)):.2f}")

model.summary()