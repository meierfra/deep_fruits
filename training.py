
import numpy as np
import load_data
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import live_plot
import matplotlib.pyplot as plt


lables = {}
X_train, Y_train_raw, inv_lables_train = load_data.load_data_cached(load_data.DATA_PATH_TRAIN, lables)
X_valid, Y_valid_raw, inv_lables_valid = load_data.load_data_cached(load_data.DATA_PATH_VAL, lables)


def convertToOneHot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='int32')
    print(np.shape(result))
    result[np.arange(len(vector)), vector] = 1
    return result


print("Y_raw:", np.shape(Y_train_raw), Y_train_raw)
print("lables: ", len(lables))

Y_train = convertToOneHot(Y_train_raw, len(lables))
Y_valid = convertToOneHot(Y_valid_raw, len(lables))


print("X shape:", np.shape(X_train))
print("Y shape:", np.shape(Y_train), "example:", Y_train[0])


def do_train(name):

    batch_size = 64
    nb_classes = len(lables)
    nb_epoch = 5
    img_rows, img_cols = 100, 100
    img_chan = 3
    kernel_size = (3, 3)
    input_shape = (img_rows, img_cols, img_chan)
    pool_size = (2, 2)

    model = Sequential()

    model.add(Convolution2D(64, kernel_size, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(64, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(128, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(128))
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
    adam = keras.optimizers.adam(lr=0.001, decay=0.003)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.summary()

    model.evaluate(X_train, Y_train)
    exit(0)

    # save model after every 10 epochs in "Checkpoints/8_facesl/model_1_fc/"-folder
    checkpoint_dir = "Checkpoints/8_faces/model_1_" + name + "/"
    print("checkpoint_dir:", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "weights_epoch_{epoch:03d}-{val_loss:.2f}.hdf5",
        verbose=1,
        save_best_only=False,
        period=10)

    # ### Training the network

    lp = live_plot.live_plot(["loss", "acc", "val_loss", "val_acc"])

    live_plot_update_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda _, logs: lp.update_points([logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"]])
    )

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=2,
              validation_data=(X_valid, Y_valid),
              # callbacks=[tensorboard, checkpointer]
              # callbacks=[live_plot_update_callback, checkpointer],
              callbacks=[live_plot_update_callback],
              )

    plt.show()



# do_train('test')
