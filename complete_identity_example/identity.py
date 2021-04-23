#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys

import datasetaccessor

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def plot_loss_curve(history):
    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.xlabel('Epoch')

    plt.show()


tf.random.set_seed(1337)

# Hyper Parameters
RANGE  = 10
REPEAT = 100
BATCH  = 10
EPOCHS = 25

TRAIN_SPLIT = 0.6
EVAL_SPLIT  = 0.2
TEST_SPLIT  = 0.2

d1 = tf.data.Dataset.range(RANGE)
d2 = tf.data.Dataset.range(RANGE)
ds = tf.data.Dataset.zip((d1, d2))

ds = ds.map(lambda x,y: (x, tf.one_hot(y, RANGE)))
ds = ds.shuffle(RANGE)
ds = ds.repeat(REPEAT)

# Split the original dataset into train, eval, and test sets
ds_size = ds.cardinality().numpy()
train_size =  ds_size * TRAIN_SPLIT
eval_size  =  ds_size * EVAL_SPLIT
test_size  =  ds_size * TEST_SPLIT

train_ds = ds.take(train_size)

eval_ds  = ds.skip(train_size)
eval_ds  = eval_ds.take(eval_size)

test_ds  = ds.skip(train_size+eval_size)
test_ds  = ds.take(test_size)

train_ds = train_ds.batch(BATCH)
eval_ds  = eval_ds.batch(BATCH)
test_ds  = test_ds.batch(BATCH)


inputs  = keras.Input(shape=(1,))
x = keras.layers.Dense(100)(inputs)
#x = keras.layers.Dense(100)(x)
#x = keras.layers.Dense(100)(x)
outputs = keras.layers.Dense(RANGE, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
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
# Just a little jank, gotta flatten out the ground truth labels from the dataset.
# Probably an easier way to do this...
test_y_hat = []
test_y     = []
for e in test_ds:
    test_y_hat.append(
        np.argmax(model.predict(e[0]), axis=1)
    )

    test_y.append(
        np.argmax(e[1].numpy(), axis=1)
    )

test_y_hat = np.ndarray.flatten(np.array(test_y_hat))
test_y     = np.ndarray.flatten(np.array(test_y))

print(test_y_hat)
print(test_y)
#confusion_ds = test_ds.unbatch().map(lambda x,y: y).batch(test_size).as_numpy_iterator()

sys.exit(1)
test_y = np.argmax(test_y, axis=1)
test_y_hat = model.predict(test_ds)
test_y_hat = np.argmax(test_y_hat, axis=1)

confusion = tf.math.confusion_matrix(test_y, test_y_hat)
plot_confusion_matrix(confusion)

derp = tf.constant([[1,],])
print(derp)
print( model(derp))
