import os
import numpy as np
import cv2

new_size = (100, 100)

def read_image(folder_path):
    images_np = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, new_size)
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            images_np.append(gray_img)
    return np.array(images_np)

img_dog_train = read_image("training_set/dogs")
img_cat_train = read_image("training_set/cats")

img_train = np.concatenate((img_dog_train, img_cat_train), axis=0)

np.random.shuffle(img_train)


