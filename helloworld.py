#! /usr/bin/python3


import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

#vdsa = datasetaccessor.VanillaDatasetAccessor(tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")
vdsa = datasetaccessor.VanillaDatasetAccessor(
    day_to_get=[2],
    #transmitter_id_to_get=[1,2],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")

# We're just gonna train an identity function
ds = vdsa.get_dataset()
#ds = ds.map(lambda inp: ([inp["transmitter_id"],], [inp["transmitter_id"],]))
ds = ds.map(lambda inp: ([inp["transmitter_id"],], [inp["transmitter_id"],])).batch(10)
#ds = ds.map(lambda inp: (inp["time_domain_IQ"], inp["transmitter_id"]))
#ds = ds.map(lambda inp: (inp["time_domain_IQ"], inp["transmitter_id"]))

#print(ds.element_spec)

inputs  = keras.Input(shape=(1,))
internal = keras.layers.Dense(100)(inputs)
outputs = keras.layers.Dense(1)(internal)

model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
model.summary()
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True)
)
model.fit(x=ds)
derp = tf.constant([[1,],])
print(derp)
print( model(derp))
