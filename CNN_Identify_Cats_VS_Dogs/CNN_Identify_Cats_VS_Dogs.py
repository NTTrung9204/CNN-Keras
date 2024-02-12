import os
import numpy as np
import cv2
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten

new_size = (124, 124)

def read_image(folder_path):
    images_np = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, new_size)
            # gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
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

# img_label_temp = one_hot_encode(2, np.array([0]*(img_dog_train.shape[0]) + [1]*(img_cat_train.shape[0])))
img_label_temp = np.array([0]*(img_dog_train.shape[0]) + [1]*(img_cat_train.shape[0]))

random_indices = np.random.permutation(img_train_temp.shape[0])

img_train = img_train_temp[random_indices]
img_label = img_label_temp[random_indices]

model = Sequential()

model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(124, 124, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(img_train, img_label, batch_size=32, epochs=15)

img_dog_pred = model.predict(img_dog_test)
img_cat_pred = model.predict(img_cat_test)

accuracy_score = 0
for result in img_dog_pred:
    if result <= 0.5 : accuracy_score+=1

for result in img_cat_pred:
    if result > 0.5 : accuracy_score+=1

accuracy_score /= (img_dog_pred.shape[0] + img_cat_pred.shape[0])

print(f"Accuracy: {100*accuracy_score:.2f}%")