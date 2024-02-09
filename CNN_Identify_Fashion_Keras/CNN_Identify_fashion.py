import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from sklearn.metrics import accuracy_score

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

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

img_training_path = "datasets/train-images-idx3-ubyte"
label_training_path = "datasets/train-labels-idx1-ubyte"
img_test_path = "datasets/t10k-images-idx3-ubyte"
label_test_path = "datasets/t10k-labels-idx1-ubyte"

img_training = load_images(img_training_path)
label_training = one_hot_encode(10, load_labels(label_training_path))
img_test = load_images(img_test_path)
label_test = one_hot_encode(10, load_labels(label_test_path))

model = Sequential()

model.add(Conv2D(32, (5, 5), activation="sigmoid", input_shape=(28, 28, 1)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), activation="sigmoid"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation="sigmoid"))

model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(img_training, label_training, batch_size=32, epochs=15)

label_prediction = model.predict(img_test)

print(f"Accuracy : {100*accuracy_score(np.argmax(label_prediction, axis=1), np.argmax(label_test, axis=1)):.2f}")

