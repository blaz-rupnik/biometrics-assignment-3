import os
import cv2
import json
from collections import Counter
import random
import numpy as np
import pickle

# constants
IMG_SIZE = 100
DATASET_PATH = "awe"

def retrieve_image_data(dir_path, image_name, img_format):
    # retrieve image and resize it
    try:
        img = cv2.imread(f"{dir_path}/{image_name}.{img_format}")
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return resized_img
    except Exception as e:
        print(f"Failed retrieving {dir_path}/{image_name}")
        return None

def retrieve_ethnicity_info(dir_path):
    # retrieve metadata and get info about ethnicity
    metadata = json.load(open(f"{dir_path}/annotations.json", encoding='utf-8'))
    return metadata['ethnicity']

data = []
X = []
y = []
samples = ['01','02','03','04','05','06','07','08','09','10']
for filename in os.listdir(DATASET_PATH):
    full_path = f"{DATASET_PATH}/{filename}"
    if os.path.isdir(full_path):
        ethnicity = retrieve_ethnicity_info(full_path)
        # one has 99, probably an error
        if ethnicity < 7:
            for sample_name in samples:
                resized_img = retrieve_image_data(full_path, sample_name, 'png')
                if resized_img is not None:
                    data.append([resized_img, ethnicity])

# for better balance reshuffle data
random.shuffle(data)
for feature, label in data:
    X.append(feature)
    y.append(label)

# reshape
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # 3 because there are 3 colour channels
# transform y to numpy array or else tensor lib has issues
y = np.array(y)

# pickle both
pickle_out = open("X_ears_eth", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y_ears_eth", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("FINISHED PREPROCESSING")