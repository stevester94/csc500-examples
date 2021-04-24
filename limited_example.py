#! /usr/bin/python3

# Toy example where we aren't even using a one hot encoding.


from steves_utils import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys

tf.random.set_seed(1337)

vdsa = datasetaccessor.SymbolDatasetAccessor(
    day_to_get=[2],
    transmitter_id_to_get=[1,2],
    #transmitter_id_to_get=[1],
    tfrecords_path="../csc500-dataset-preprocessor/symbol_tfrecords/")


ds = vdsa.get_dataset()
ds = ds.map(lambda inp: ( inp["frequency_domain_IQ"], inp["transmitter_id"]))
#ds = ds.take(100)
ds = ds.batch(100)

# Note that even with this input shape, keras still expects the input to be batched
inputs  = keras.Input(shape=(2,48))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(150)(x)
x = keras.layers.Dense(150)(x)
x = keras.layers.Dense(150)(x)
x = keras.layers.Dense(150)(x)
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())
              #metrics=['accuracy'])

model.fit(x=ds, epochs=10)
#derp = tf.constant([[1,],])
#print(derp)
#print( model(derp))
