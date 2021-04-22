#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys

tf.random.set_seed(1337)

# Hyper Parameters
RANGE  = 10
REPEAT = 100
BATCH  = 10
EPOCHS = 50

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
train_ds = ds.take(ds.cardinality().numpy() * TRAIN_SPLIT)

eval_ds  = ds.skip(train_ds.cardinality())
eval_ds  = eval_ds.take(ds.cardinality().numpy() * EVAL_SPLIT)

test_ds  = ds.skip(train_ds.cardinality() + eval_ds.cardinality())
eval_ds  = eval_ds.take(train_ds.cardinality().numpy() *  TEST_SPLIT)

train_ds = train_ds.batch(BATCH)
eval_ds  = eval_ds.batch(BATCH)
test_ds  = test_ds.batch(BATCH)

for e in test_ds:
    print(e)


#sys.exit(1)

ds = ds.batch(BATCH)

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
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="model_checkpoint",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
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

derp = tf.constant([[1,],])
print(derp)
print( model(derp))