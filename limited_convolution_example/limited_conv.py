#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys
from datetime import datetime

from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve
from steves_utils import utils
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Setting the seed is vital for reproducibility
    tf.random.set_seed(1337)

    # Hyper Parameters
    RANGE   = len(ALL_SERIAL_NUMBERS)
    EPOCHS  = 2
    DROPOUT = 0.5 # [0,1], the chance to drop an input
    BATCH = 1000

    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)

    
    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        runs_to_get=[1],
        distances_to_get=[8],
        serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
    )

    print("Total Examples:", cardinality)
    print("That's {}GB of data (at least)".format( cardinality * ORIGINAL_PAPER_SAMPLES_PER_CHUNK * 2 * 8 / 1024 / 1024 / 1024))

    input("Press enter to continue calmly")

    num_train = int(cardinality * TRAIN_SPLIT)
    num_val = int(cardinality * VAL_SPLIT)
    num_test = int(cardinality * TEST_SPLIT)

    ds = ds.shuffle(cardinality)

    train_ds = ds.take(num_train)
    val_ds = ds.skip(num_train).take(num_val)
    test_ds = ds.skip(num_train+num_val).take(num_test)
    

    # train_ds = train_ds.unbatch().filter(lambda freq_iq, day, transmitter_id, transmission_id, symbol_index_in_file: day == 1).batch(100)
    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
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

    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)


    train_ds = train_ds.cache().shuffle(num_train, reshuffle_each_iteration=True)
    val_ds   = val_ds.cache()
    test_ds  = test_ds.cache()

    # for e in train_ds.unbatch():
    #     print(e)

    inputs  = keras.Input(shape=(2,ORIGINAL_PAPER_SAMPLES_PER_CHUNK))

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
