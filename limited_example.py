#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys

tf.random.set_seed(1337)

vdsa = datasetaccessor.SymbolDatasetAccessor(
    day_to_get=[2],
    transmitter_id_to_get=[1,2],
    tfrecords_path="../csc500-dataset-preprocessor/symbol_tfrecords/")


ds = vdsa.get_dataset()
ds = ds.map(lambda inp: ( inp["frequency_domain_IQ"], inp["transmitter_id"]))
ds = ds.batch(100)


inputs  = keras.Input(shape=(2,48))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(96)(x)
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
