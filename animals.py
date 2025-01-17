import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/blazr/OneDrive/Namizje/fri_20_21/Slikovna_Biometrija/Tutorials/Assignment3/biometrics-assignment-3/animals"
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 50
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because it is greyscale, 3 if coloured

pickle_out = open("X_animals.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_animals.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#pickle_in = open("X.pickle", "rb") for reading
#X = pickle.load(pickle_in)