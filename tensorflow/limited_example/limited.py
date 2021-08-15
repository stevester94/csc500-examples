#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys

from steves_utils import datasetaccessor
from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from steves_utils.datasetaccessor import SymbolDatasetAccessor


# Setting the seed is vital for reproducibility
tf.random.set_seed(1337)

# Hyper Parameters
RANGE  = 12
BATCH  = 200
EPOCHS = 100

TRAIN_SPLIT = 0.6
EVAL_SPLIT  = 0.2
TEST_SPLIT  = 0.2

#ds_size = 2233502 # Too lazy, lets goooo
# dsa = datasetaccessor.SymbolDatasetAccessor(
#     day_to_get=[1,2],
#     transmitter_id_to_get=[10,11],
#     transmission_id_to_get=[1,2],
#     tfrecords_path="../../csc500-dataset-preprocessor/symbol_tfrecords/")

dsa = datasetaccessor.SymbolDatasetAccessor(
    day_to_get=[1],
    transmitter_id_to_get=[10,11],
    transmission_id_to_get=[1],
    tfrecords_path="../../csc500-dataset-preprocessor/symbol_tfrecords/")

ds = dsa.get_dataset()
    

# Split the original dataset into train, eval, and test sets
#ds_size = dsa.get_dataset_cardinality()
ds_size = 515368
print("cardinality: ", ds_size)

# Times
# TDLR: batching is good, prefetching doesn't help at least with the tight iterative loop I used.
# Splitting into datasets then batching slows down some but not too much
#
# Dataset:
#     day_to_get=[1,2],
#     transmitter_id_to_get=[10,11],
#     transmission_id_to_get=[1,2],
#
# Actual Times:
# Naive one by one mapping: 2m4.322s, 80% Idle
# Map then batch 200: 0m30.649s, 40% Idle
# Batch then map 200: 0m24.497s, 40% Idle
# Batch then map 200 then prefetch 5: 0m24.661s, 40% Idle
# Map then batch via the split sets: 0m42.773s, 48% Idle
ds = ds.map(lambda inp: ( inp["frequency_domain_IQ"], inp["transmitter_id"]))
ds = ds.map(lambda x,y: (x, tf.one_hot(y, RANGE)))

train_size =  int(ds_size * TRAIN_SPLIT)
eval_size  =  int(ds_size * EVAL_SPLIT)
test_size  =  int(ds_size * TEST_SPLIT)

train_ds = ds.take(train_size)

eval_ds  = ds.skip(train_size)
eval_ds  = eval_ds.take(eval_size)

test_ds  = ds.skip(train_size+eval_size)
test_ds  = ds.take(test_size)

train_ds = train_ds.batch(BATCH)
eval_ds  = eval_ds.batch(BATCH)
test_ds  = test_ds.batch(BATCH)

inputs  = keras.Input(shape=(2,48))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(100)(x)
x = keras.layers.Dense(100)(x)
x = keras.layers.Dense(100)(x)
outputs = keras.layers.Dense(RANGE, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(), # This may do better with categorical_crossentropy
              metrics=[keras.metrics.CategoricalAccuracy()], # Categorical is needed for one hot encoded data
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved at the end of this epoch
        # Note that this is a dir
        filepath="model_checkpoint",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss", # We could theoretically monitor the eval loss as well (eh)
        verbose=0,
    )
]

#  History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, 
#    as well as validation loss values and validation metrics values (if applicable).
history = model.fit(
    x=train_ds, 
    epochs=EPOCHS,
    validation_data=eval_ds,
    callbacks=callbacks,
)

print("Now we evaluate on the test data")
results = model.evaluate(test_ds)
print("test loss:", results[0], ", test acc:", results[1])

# Now we generate the confusion matrix and loss graph

# Confusion matrix
# It's jank because we have a dataset of tuples, and there's not a great way to separate the halves without iterating twice.
# However, iterating twice will not work because our dataset is shuffled on each iteration.
test_y_hat = []
test_y     = []

# This is actually very slow
for e in test_ds:
    test_y_hat.extend(
        list(np.argmax(model.predict(e[0]), axis=1))
    )

    test_y.extend(
        list(np.argmax(e[1].numpy(), axis=1))
    )


#test_y_hat = np.ndarray.flatten(np.array(test_y_hat))
#test_y     = np.ndarray.flatten(np.array(test_y))

# I've checked both of these calls, they work correctly
confusion = tf.math.confusion_matrix(test_y, test_y_hat)
#plot_confusion_matrix(confusion)
save_confusion_matrix(confusion)


# Loss curve
#plot_loss_curve(history)
save_loss_curve(history)