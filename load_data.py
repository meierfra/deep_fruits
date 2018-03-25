from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


DATA_PATH_BASE = "./data/fruits-360"
DATA_PATH_TRAIN = DATA_PATH_BASE + "/Training"
DATA_PATH_VAL = DATA_PATH_BASE + "/Training"


def load_image(image_path):
    with Image.open(image_path) as img:
        img.load()
        data = np.asarray(img, dtype="uint8")
        return data


def load_data(path, lables):
    inv_lables = {}
    class_idx = 0
    X = []
    Y = []

    wd1 = path
    for class_name in os.listdir(wd1):
        class_idx = lables.get(class_name)
        if not class_idx:
            class_idx = len(lables)
            lables.update({class_name: class_idx})

        inv_lables.update({class_idx: class_name})
        wd2 = wd1 + "/" + class_name
        count = 0
        for image_name in os.listdir(wd2):
            count += 1
            image_path = wd2 + "/" + image_name
            data = load_image(image_path)
            X.append(data)
            Y.append(class_idx)

        print(class_name, class_idx, count)
        lables.update({class_idx: class_name})
        class_idx += 1
    return X, Y, inv_lables


TRAIN_DATA_FIlE = './data/fruits_train.pickle'

if not os.path.isfile(TRAIN_DATA_FIlE):
    lables_index = {}
    X_train, Y_train, inv_lables_train = load_data(DATA_PATH_TRAIN, lables_index)
    data = [X_train, Y_train, inv_lables_train]
    with open(TRAIN_DATA_FIlE, 'wb') as f:
        pickle.dump(data, f)
else:
    with open(TRAIN_DATA_FIlE, 'rb') as f:
        data = pickle.load(f)
        X_train, Y_train, inv_lables_train = data


r = np.random.randint(0, len(X_train))
plt.imshow(np.asarray(X_train[r], dtype="uint8"), interpolation="bicubic")
print("r:", r, "Y:", Y_train[r], "lablel:", inv_lables_train.get(Y_train[r]))
plt.show()
