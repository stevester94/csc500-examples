#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys
from datetime import datetime

from steves_utils import datasetaccessor
from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve
from steves_utils.datasetaccessor import SymbolDatasetAccessor
from steves_utils.binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor
from steves_utils.binary_symbol_dataset_accessor import BinarySymbolDatasetAccessor

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np



# Setting the seed is vital for reproducibility
tf.random.set_seed(1337)

# Hyper Parameters
RANGE   = 12
BATCH   = 100
EPOCHS  = 5
DROPOUT = 0.5 # [0,1], the chance to drop an input

files = [
    "../../csc500-dataset-preprocessor/bin/day-1_transmitter-10_transmission-1.bin",
    "../../csc500-dataset-preprocessor/bin/day-1_transmitter-11_transmission-1.bin",
]

bsda = BinarySymbolDatasetAccessor(
    seed=1337,
    batch_size=BATCH,
    num_class_labels=RANGE,
    bin_path="../../csc500-dataset-preprocessor/bin/",
    day_to_get=[1],
    transmitter_id_to_get=[10,11],
    transmission_id_to_get=[1],
)


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
        monitor="val_loss", # We could theoretically monitor the eval loss as well (eh)
        verbose=0,
    ),
    keras.callbacks.TensorBoard(log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
]

history = model.fit(
    x=bsda.get_train_generator(),
    # batch_size=BATCH,
    steps_per_epoch=int(bsda.get_train_dataset_cardinality()/BATCH),
    epochs=EPOCHS,
    validation_data=bsda.get_eval_generator(),
    validation_steps=int(bsda.get_eval_dataset_cardinality()/BATCH),
    # callbacks=callbacks,
)

print("Now we evaluate on the test data")
results = model.evaluate(bsda.get_test_generator())
print("test loss:", results[0], ", test acc:", results[1])

test_y_hat = []
test_y     = []

# This is actually very slow
for e in bsda.get_test_generator():
    test_y_hat.extend(
        list(np.argmax(model.predict(e[0]), axis=1))
    )

    test_y.extend(
        list(np.argmax(e[1].numpy(), axis=1))
    )

confusion = tf.math.confusion_matrix(test_y, test_y_hat)
#plot_confusion_matrix(confusion)
save_confusion_matrix(confusion)


# Loss curve
#plot_loss_curve(history)
save_loss_curve(history)