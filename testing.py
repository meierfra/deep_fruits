
import numpy as np
import load_data
# import os

import keras
import matplotlib.pyplot as plt


def plot_pics(imgs):
    nimgs = len(imgs)
    #plt.figure(figsize=(nimgs // 10 + 1, nimgs % 10))
    plt.figure(figsize=(10,10))
    for i in range(0, nimgs):
        plt.subplot(nimgs // 10 + 1, nimgs % 10, (i + 1))
        plt.imshow(np.asarray(imgs[i], dtype="uint8"), interpolation="bicubic")
        plt.axis('off')
        # plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()



lables = {}
inv_lables = {}
X_train_raw, Y_train_raw, inv_lables_train = load_data.load_data_cached(load_data.DATA_PATH_TRAIN, lables)
X_valid_raw, Y_valid_raw, inv_lables_valid = load_data.load_data_cached(load_data.DATA_PATH_VAL, lables)
inv_lables.update(inv_lables_train)
inv_lables.update(inv_lables_valid)

X_mean = np.mean(X_train_raw, axis=0)
X_std = np.std(X_train_raw, axis=0)

imgs_raw = []


r = np.random.randint(0, len(X_valid_raw))
print("random image: class: " + str(Y_valid_raw[r]) + ": " + inv_lables[Y_valid_raw[r]])
imgs_raw.append(X_valid_raw[r])

imgs_raw.append(load_data.load_image('./data/240px-Honeycrisp.jpg', (100, 100)))
imgs_raw.append(load_data.load_image('./data/Banana-Single.jpg', (100, 100)))
print(np.shape(imgs_raw[0]))

plot_pics(imgs_raw)


imgs = (imgs_raw - X_mean) / (X_std + 0.0001)

#model = keras.models.load_model('./Checkpoints/model_cnn_large_aug_dout_bn_test1/weights_epoch_010-0.02.hdf5')
model = keras.models.load_model('./Checkpoints/model_cnn_simple_test2/weights_epoch_025-0.03.hdf5')

preds = model.predict(imgs)

#preds = np.array([6.0, 5.0, 1.0, 10.0]).reshape(1, 4)
print(np.shape(preds))

for pred_num, pred in enumerate(preds):
    print("-----" + str(pred_num) + "-----------------------------")
    print(list(enumerate(pred)))

    pred_idx = np.array([(x, y) for x, y in enumerate(pred)])
    pred_idx_sorted = pred_idx[pred_idx[:, 1].argsort()[::-1]]

    for e in pred_idx_sorted[:5]:
        class_idx, propability = int(e[0]), float(e[1])
        class_name = inv_lables.get(class_idx)
        print("image is of class: " + str(class_idx) + ": '" + class_name + "' with propability of " + str(propability))

plt.show()
