import os
import numpy as np
import cv2
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.applications import DenseNet121

new_size = (224, 224)

def read_image(folder_path):
    images_np = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, new_size)
            images_np.append(resized_img)
    return np.array(images_np)

def one_hot_encode(num_classes, input_values):
    encoded_values = np.zeros((len(input_values), num_classes))
    for i, value in enumerate(input_values):
        encoded_values[i, value] = 1
    return encoded_values

img_dog_train = read_image("training_set/dogs")
img_cat_train = read_image("training_set/cats")

img_dog_test = read_image("test_set/dogs")
img_cat_test = read_image("test_set/cats")

img_train_temp = np.concatenate((img_dog_train, img_cat_train), axis=0)

img_label_temp = np.array([0]*(img_dog_train.shape[0]) + [1]*(img_cat_train.shape[0]))

random_indices = np.random.permutation(img_train_temp.shape[0])

img_train = img_train_temp[random_indices]
img_label = img_label_temp[random_indices]

# Khởi tạo mô hình DenseNet121 với trọng số được tải sẵn từ ImageNet
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng các lớp trong mô hình DenseNet121
for layer in densenet_model.layers:
    layer.trainable = False

# Xây dựng mô hình phân loại bằng Sequential API
model = Sequential()

# Thêm mô hình DenseNet121 vào mô hình Sequential
model.add(densenet_model)

# Thêm lớp Flatten để chuyển từ tensor 3D thành vector 1D
model.add(Flatten())

# Thêm lớp fully connected với 256 đơn vị và hàm kích hoạt ReLU
model.add(Dense(256, activation='relu'))

# Thêm lớp đầu ra với 2 đơn vị và hàm kích hoạt softmax cho phân loại chó/mèo
model.add(Dense(1, activation='sigmoid'))

# Biên soạn mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(img_train, img_label, batch_size=32, epochs=5)

img_dog_pred = model.predict(img_dog_test)
img_cat_pred = model.predict(img_cat_test)

accuracy_score = 0
for result in img_dog_pred:
    if result <= 0.5 : accuracy_score+=1

for result in img_cat_pred:
    if result > 0.5 : accuracy_score+=1

accuracy_score /= (img_dog_pred.shape[0] + img_cat_pred.shape[0])

print(f"Accuracy: {100*accuracy_score:.2f}%")

model.summary()

model.save('Model_DenseNet121.keras')