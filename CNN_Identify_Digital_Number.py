import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from sklearn.metrics import accuracy_score

def one_hot_encode(num_classes, input_values):
    """
    Mã hóa one-hot cho một mảng numpy chứa các giá trị đầu vào.

    Parameters:
    num_classes (int): Số lượng lớp phân loại.
    input_values (numpy.ndarray): Mảng numpy chứa các giá trị đầu vào.

    Returns:
    numpy.ndarray: Mảng numpy biểu diễn mã hóa one-hot của các giá trị đầu vào.
    """
    encoded_values = np.zeros((len(input_values), num_classes))  # Tạo mảng kết quả với kích thước phù hợp

    for i, value in enumerate(input_values):
        encoded_values[i, value] = 1  # Đặt giá trị 1 tại chỉ mục tương ứng với giá trị đầu vào

    return encoded_values

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
train_labels = one_hot_encode(10, train_labels)

test_images = test_images.reshape(10000, 28, 28, 1)
test_labels = one_hot_encode(10, test_labels)

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='sigmoid', input_shape=(28, 28, 1)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), activation='sigmoid'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='sigmoid'))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=10)

test_predict = model.predict(test_images)

accuracy = 0
for i in range(0, 10000):
    if(np.argmax(test_labels[i]) == np.argmax(test_predict[i])):
        accuracy += 1

print("accuracy: ", accuracy)

for i in range(20):
    img = cv2.imread(f'image/my_image{i}.png',0).reshape(1, 28, 28, 1)
    result = model.predict(img)
    print(i, np.argmax(result))