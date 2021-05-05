#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys
from datetime import datetime

from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve
from steves_utils import utils

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Setting the seed is vital for reproducibility
    tf.random.set_seed(1337)

    # Hyper Parameters
    RANGE   = 20 + 1 #We have 20 transmitters. one hot is 0 indexed but our transmitters are 1 indexed.
    RECORD_BATCH   = 100 # NOTE: This is how many records are batched in our binary files, not how many we want in our actual batch
    EPOCHS  = 10
    DROPOUT = 0.5 # [0,1], the chance to drop an input

    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)

    
    accessor = utils.shuffled_dataset_accessor(
        path="../../csc500-dataset-preprocessor/day-1_ultra-limited",
        record_batch_size=1000,
        desired_batch_size=100
        # train_val_test_split = (1, 0, 0)
    )

    train_ds = accessor["train_ds"]
    test_ds = accessor["test_ds"]
    val_ds = accessor["val_ds"]
    total_records = accessor["total_records"]


    print(total_records)

    # print("WE ARE TRAINING AGAINST DAY")
    # input("Press enter to continue calmly")

    # train_ds = train_ds.unbatch().filter(lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: day == 1).batch(100)
    train_ds = train_ds.map(
        lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: (freq_iq, tf.one_hot(transmitter_id, RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: (freq_iq, tf.one_hot(transmitter_id, RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: (freq_iq, tf.one_hot(transmitter_id, RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # train_ds = train_ds.map(
    #     lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: (freq_iq, transmitter_id),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # val_ds = val_ds.map(
    #     lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: (freq_iq, transmitter_id),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # test_ds = test_ds.map(
    #     lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: (freq_iq, transmitter_id),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # train_ds = train_ds.prefetch(100).unbatch().batch(5000)
    # train_ds = train_ds.prefetch(100).take(2)
    # val_ds   = val_ds.prefetch(100).take(2)
    # test_ds  = test_ds.prefetch(100).take(2)

    # train_ds = train_ds.prefetch(100).take(1).cache().repeat(100000).repeat(100000)
    # train_ds = train_ds.batch(100).map(lambda x,y: (tf.reshape(x, [10000, 2, 48]), tf.reshape(y, [10000, 21]))).prefetch(100)
    # train_ds = train_ds.caprefetch(100)
    # val_ds   = val_ds.prefetch(100)
    # test_ds  = test_ds.prefetch(100)

    train_ds = train_ds.unbatch().batch(10000).cache().shuffle(total_records, reshuffle_each_iteration=True).repeat(100)
    val_ds   = val_ds.cache()
    test_ds  = test_ds.cache()

    # for e in train_ds.unbatch():
    #     print(e)

    inputs  = keras.Input(shape=(2,48))

    x = keras.layers.Convolution1D(
        filters=50,
        kernel_size=7,
        strides=1,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first",
        name="classifier_3"
    )(inputs)

    x = keras.layers.Convolution1D(
        filters=50,
        kernel_size=7,
        strides=2,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first",
        name="classifier_4"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    x = keras.layers.Flatten(name="classifier_5")(x)

    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_6"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    x = keras.layers.Dense(
        units=80,
        activation='relu',
        kernel_initializer='he_normal',
        name="classifier_7"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    outputs = keras.layers.Dense(RANGE, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(), # This may do better with categorical_crossentropy
                metrics=[keras.metrics.CategoricalAccuracy()], # Categorical is needed for one hot encoded data
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="model_checkpoint",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss", # We could theoretically monitor the val loss as well (eh)
            verbose=0,
        ),
        # keras.callbacks.TensorBoard(log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    ]

    history = model.fit(
        x=train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )


    print("Now we evaluate on the test data")
    results = model.evaluate(
        test_ds,
        verbose=1,
    )
    print("test loss:", results[0], ", test acc:", results[1])

    test_y_hat = []
    test_y     = []

    print("Calculate the confusion matrix")
    total_confusion = None
    f = None
    for e in test_ds:
        confusion = tf.math.confusion_matrix(
            np.argmax(model.predict(e[0]), axis=1),
            np.argmax(e[1].numpy(), axis=1),
            num_classes=RANGE
        )

        if total_confusion == None:
            total_confusion = confusion
        else:
            total_confusion = total_confusion + confusion

    #plot_confusion_matrix(confusion)
    save_confusion_matrix(confusion)


    # Loss curve
    #plot_loss_curve(history)
    save_loss_curve(history)
