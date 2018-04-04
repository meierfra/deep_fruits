from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


DATA_PATH_BASE = "./data/fruits-360"
DATA_PATH_TRAIN = DATA_PATH_BASE + "/Training"
DATA_PATH_VAL = DATA_PATH_BASE + "/Validation"
DATA_PATH_MULTI = DATA_PATH_BASE + "/test-multiple_fruits"


def load_image(image_path, rescale_dim=None):
    with Image.open(image_path) as img:
        img.load()
        if (rescale_dim):
            img = ImageOps.fit(img, rescale_dim)
        data = np.asarray(img, dtype="uint8")
        return data


def load_data_path(path, lables):
    inv_lables = {}
    class_idx = 0
    X = []
    Y = []

    wd1 = path
    for class_name in os.listdir(wd1):
        class_idx = lables.get(class_name)
        if class_idx is None:
            class_idx = len(lables)
            lables.update({class_name: class_idx})

        wd2 = wd1 + "/" + class_name
        count = 0
        for image_name in os.listdir(wd2):
            count += 1
            image_path = wd2 + "/" + image_name
            data = load_image(image_path)
            X.append(data)
            Y.append(class_idx)

        print(class_name, class_idx, count)
        inv_lables.update({class_idx: class_name})
    return X, Y, inv_lables


def load_data_cached(src_path, lables):
    pickle_file = src_path + ".pickle"
    if not os.path.isfile(pickle_file):
        X, Y, inv_lables = load_data_path(src_path, lables)
        data = [X, Y, inv_lables, lables]
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        print("load cached data from:", pickle_file)
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            X, Y, inv_lables, lables_tmp = data
            lables.update(lables_tmp)
    return X, Y, inv_lables


if __name__ == "__main__":
    lables = {}
    X_train, Y_train, inv_lables_train = load_data_cached(DATA_PATH_TRAIN, lables)
    X_valid, Y_valid, inv_lables_valid = load_data_cached(DATA_PATH_VAL, lables)
    print("#lables: ", len(inv_lables_train), len(lables))
    print("lables:", lables)
    print(inv_lables_train)
    print(lables)
    print("X shape:", np.shape(X_train))
    print("Y shape:", np.shape(Y_train), "example:", Y_train[0])

    r = np.random.randint(0, len(X_train))
    plt.imshow(np.asarray(X_train[r], dtype="uint8"), interpolation="bicubic")
    print("r:", r, "Y:", Y_train[r], "lablel:", inv_lables_train.get(Y_train[r]))
    plt.show()
