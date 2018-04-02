
import numpy as np
import load_data
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import live_plot
import matplotlib.pyplot as plt


def convertToOneHot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='int32')
    print(np.shape(result))
    result[np.arange(len(vector)), vector] = 1
    return result


X_train = None
Y_train = None
X_valid = None
Y_valid = None
lables = {}
inv_lables = {}


def prepare_data():
    global X_train, Y_train, X_valid, Y_valid, lables, inv_lables

    X_train_raw, Y_train_raw, inv_lables_train = load_data.load_data_cached(load_data.DATA_PATH_TRAIN, lables)
    X_valid_raw, Y_valid_raw, inv_lables_valid = load_data.load_data_cached(load_data.DATA_PATH_VAL, lables)
    inv_lables.update(inv_lables_train)
    inv_lables.update(inv_lables_valid)

#     print("Xr shape:", np.shape(X_train_raw))
#     print("Xr example:", X_train_raw[0])
#     print("Yr shape:", np.shape(Y_train_raw))
#     print("Yr example:", Y_train_raw[0])

    # #### Normalization of the training and validationset.
    X_mean = np.mean(X_train_raw, axis=0)
    X_std = np.std(X_train_raw, axis=0)

    # normalize data through pixel axis
    X_train = (X_train_raw - X_mean) / (X_std + 0.0001)
    X_valid = (X_valid_raw - X_mean) / (X_std + 0.0001)

    # normalize each (channel of each) picture
#     x_mean = np.ndarray(3)
#     x_std = np.ndarray(3)
#     for i in range(3):
#         x_mean[i] = np.mean(X_train[:, :, :, i])
#         x_std[i] = np.std(X_train[:, :, :, i]) + 0.0001
#     print(x_mean, x_std)
#
#     X_train = X_train - np.full((48, 48, 3), x_mean)
#     X_train = X_train / np.full((48, 48, 3), x_std)
#
#     X_valid = X_valid - np.full((48, 48, 3), x_mean)
#     X_valid = X_valid / np.full((48, 48, 3), x_std)

    Y_train = convertToOneHot(Y_train_raw, len(lables))
    Y_valid = convertToOneHot(Y_valid_raw, len(lables))

#     print("X shape:", np.shape(X_train))
#     print("X example:", X_train[0])
#     print("Y shape:", np.shape(Y_train))
#     print("Y example:", Y_train[0])


def do_train():

    batch_size = 128
    nb_classes = len(lables)
    nb_epoch = 150
    img_rows, img_cols = 100, 100
    img_chan = 3
    kernel_size = (3, 3)
    input_shape = (img_rows, img_cols, img_chan)
    pool_size = (2, 2)

    name = 'cnn_simple_test2'
    model = Sequential()

    model.add(Convolution2D(32, kernel_size, padding='valid', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#     model.add(Convolution2D(64, kernel_size, padding='valid'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
#
    model.add(Convolution2D(32, kernel_size, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#     model.add(Convolution2D(64, kernel_size, padding='valid'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
#
    model.add(Convolution2D(32, kernel_size, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # adam = keras.optimizers.adam()
    adam = keras.optimizers.adam(lr=0.001, decay=0.000)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.summary()
    checkpoint_dir = "Checkpoints/model_" + name + "/"
    print("checkpoint_dir:", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    keras.utils.plot_model(model, show_shapes=True, to_file=checkpoint_dir + 'model.png')

#     model.evaluate(X_train, Y_train)
#     exit(0)

    # save model after every 5 epochs in "Checkpoints/8_facesl/model_1_fc/"-folder
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "weights_epoch_{epoch:03d}-{val_loss:.3f}.hdf5",
        verbose=1,
        save_best_only=False,
        period=5)

    # ### Training the network

    lp = live_plot.live_plot(["loss", "acc", "val_loss", "val_acc"])

    def live_plot_update_and_save(epoch, logs):
        lp.update_points([logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"]])
        lp.save(checkpoint_dir + "loss_acc_plot_epoch{:03d}.png".format(epoch))

    live_plot_update_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=live_plot_update_and_save
    )

#     datagen = ImageDataGenerator(rotation_range=90,
#                                  width_shift_range=0.1,
#                                  height_shift_range=0.1,
#                                  horizontal_flip=True,
#                                  vertical_flip=True,
#                                  zoom_range=0.1)
#
#     train_gen = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)

    # model.fit_generator(train_gen,
    model.fit(X_train, Y_train, batch_size=batch_size,
              epochs=nb_epoch,
              verbose=2,
              validation_data=(X_valid, Y_valid),
              # callbacks=[tensorboard, checkpointer]
              callbacks=[live_plot_update_callback, checkpointer],
              # callbacks=[live_plot_update_callback],
              )

    plt.show()


prepare_data()
do_train()
