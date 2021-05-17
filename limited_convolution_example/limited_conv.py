#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys, os

from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

def get_all_shuffled():
    from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
    from steves_utils import utils

    RANGE   = len(ALL_SERIAL_NUMBERS)
    path = os.path.join(utils.get_datasets_base_path(), "all_shuffled", "output")
    print(utils.get_datasets_base_path())
    print(path)
    datasets = Shuffled_Dataset_Factory(path, train_val_test_splits=(0.6, 0.2, 0.2))

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]    

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

    return train_ds, val_ds, test_ds

# This works
def get_limited_oracle():
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)
    BATCH=1000
    RANGE   = len(ALL_SERIAL_NUMBERS)

    
    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        runs_to_get=[1],
        distances_to_get=[8],
        serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
    )

    print("Total Examples:", cardinality)
    print("That's {}GB of data (at least)".format( cardinality * ORIGINAL_PAPER_SAMPLES_PER_CHUNK * 2 * 8 / 1024 / 1024 / 1024))

    num_train = int(cardinality * TRAIN_SPLIT)
    num_val = int(cardinality * VAL_SPLIT)
    num_test = int(cardinality * TEST_SPLIT)

    ds = ds.shuffle(cardinality)

    train_ds = ds.take(num_train)
    val_ds = ds.skip(num_train).take(num_val)
    test_ds = ds.skip(num_train+num_val).take(num_test)

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

    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)

    return train_ds, val_ds, test_ds

def get_less_limited_oracle():
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)
    BATCH=1000
    RANGE   = len(ALL_SERIAL_NUMBERS)

    
    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        runs_to_get=[1],
        distances_to_get=ALL_DISTANCES_FEET[:1],
        # serial_numbers_to_get=ALL_SERIAL_NUMBERS[:6]
    )

    print("Total Examples:", cardinality)
    print("That's {}GB of data (at least)".format( cardinality * ORIGINAL_PAPER_SAMPLES_PER_CHUNK * 2 * 8 / 1024 / 1024 / 1024))
    input("Pres Enter to continue")
    num_train = int(cardinality * TRAIN_SPLIT)
    num_val = int(cardinality * VAL_SPLIT)
    num_test = int(cardinality * TEST_SPLIT)

    ds = ds.shuffle(cardinality)
    ds = ds.cache("/tmp/muh_cache")

    # Prime the cache
    for e in ds.batch(1000):
        pass

    # print("Comment this out next time")
    # sys.exit(1)

    train_ds = ds.take(num_train)
    val_ds = ds.skip(num_train).take(num_val)
    test_ds = ds.skip(num_train+num_val).take(num_test)

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
    
    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    # Setting the seed is vital for reproducibility
    tf.random.set_seed(1337)

    # Hyper Parameters
    RANGE   = len(ALL_SERIAL_NUMBERS)
    EPOCHS  = 200
    DROPOUT = 0.5 # [0,1], the chance to drop an input


    # train_ds, val_ds, test_ds = get_all_shuffled()
    train_ds, val_ds, test_ds = get_less_limited_oracle()

    # train_ds = train_ds.unbatch().batch(1).take(1).cache().prefetch(100)
    # val_ds   = val_ds.unbatch().batch(1).take(1).cache().prefetch(100)
    # test_ds  = test_ds.unbatch().batch(1).take(1).cache().prefetch(100)

    # train_ds = train_ds.take(10).cache().prefetch(100)
    # val_ds   = val_ds.take(2).cache().prefetch(100)
    # test_ds  = test_ds.take(2).cache().prefetch(100)

    # train_ds = train_ds.cache().prefetch(100)
    # val_ds   = val_ds.cache().prefetch(100)
    # test_ds  = test_ds.cache().prefetch(100)

    # for e in train_ds.unbatch():
    #     print( e[1].numpy() )

    # sys.exit(1)

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
            np.argmax(e[1].numpy(), axis=1),
            np.argmax(model.predict(e[0]), axis=1),
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
