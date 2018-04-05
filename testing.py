
import numpy as np
import load_data
# import os

import keras
import matplotlib.pyplot as plt


def plot_pics(imgs):
    nimgs = len(imgs)
    #plt.figure(figsize=(nimgs // 10 + 1, nimgs % 10))
    plt.figure(figsize=(10, 10))
    for i in range(0, nimgs):
        plt.subplot(nimgs // 10 + 1, nimgs % 10, (i + 1))
        plt.imshow(np.asarray(imgs[i], dtype="uint8"), interpolation="bicubic")
        plt.axis('off')
        # plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # plt.show()


def convertToOneHot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='int32')
    print(np.shape(result))
    result[np.arange(len(vector)), vector] = 1
    return result


lables = {}
inv_lables = {}
X_train_raw, Y_train_raw, inv_lables_train = load_data.load_data_cached(load_data.DATA_PATH_TRAIN, lables)
X_valid_raw, Y_valid_raw, inv_lables_valid = load_data.load_data_cached(load_data.DATA_PATH_VAL, lables)
inv_lables.update(inv_lables_train)
inv_lables.update(inv_lables_valid)

# Normalization of the training and validationset.
X_mean = np.mean(X_train_raw, axis=0)
X_std = np.std(X_train_raw, axis=0)

#X_train = (X_train_raw - X_mean) / (X_std + 0.0001)
#X_valid = (X_valid_raw - X_mean) / (X_std + 0.0001)
#Y_train = convertToOneHot(Y_train_raw, len(lables))
#Y_valid = convertToOneHot(Y_valid_raw, len(lables))

# normalize each (channel of each) picture
# img_rows, img_cols = 100, 100
# img_chan = 3
# input_shape = (img_rows, img_cols, img_chan)
# x_mean = np.ndarray(img_chan)
# x_std = np.ndarray(img_chan)
# for i in range(img_chan):
#     x_mean[i] = np.mean(np.array(X_train_raw)[:, :, :, i])
#     x_std[i] = np.std(np.array(X_train_raw)[:, :, :, i]) + 0.0001

imgs_raw = []


r = np.random.randint(0, len(X_valid_raw))
print("random image: class: " + str(Y_valid_raw[r]) + ": " + inv_lables[Y_valid_raw[r]])
imgs_raw.append(X_valid_raw[r])

imgs_raw.append(load_data.load_image('./data/240px-Honeycrisp.jpg', (100, 100)))
imgs_raw.append(load_data.load_image('./data/Banana-Single.jpg', (100, 100)))
print(np.shape(imgs_raw[0]))

plot_pics(imgs_raw)


imgs = (imgs_raw - X_mean) / (X_std + 0.0001)
# imgs = (imgs_raw - np.full(input_shape, x_mean)) / np.full(input_shape, x_std)


#model = keras.models.load_model('./Checkpoints/model_cnn_simple_test2/weights_epoch_025-0.03.hdf5')
#model = keras.models.load_model('./Checkpoints/model_cnn_large_bn_dout_test2/weights_epoch_020-0.012.hdf5')
model = keras.models.load_model('./Checkpoints/model_cnn_large_aug_bn_dout_test3/weights_epoch_085-0.053.hdf5')
model.summary()

# check model on train data
#print("CHECK MODEL:")
#preds = model.predict(X_train)
#print("Acc = ", np.sum(np.argmax(Y_train, axis=1) == np.argmax(preds, axis=1)) / len(preds))
#preds = model.predict(X_valid)
#print("VAL_Acc = ", np.sum(np.argmax(Y_valid, axis=1) == np.argmax(preds, axis=1)) / len(preds))


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
