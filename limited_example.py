#! /usr/bin/python3

import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys

tf.random.set_seed(1337)

vdsa = datasetaccessor.VanillaDatasetAccessor(
    day_to_get=[2],
    transmitter_id_to_get=[1,2],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")


nested = [[1, 2, 3, 4], [5, 6, 7, 8]]
ds = tf.data.Dataset.from_tensor_slices(nested)

print(ds.element_spec)

sys.exit(1)

ds = vdsa.get_dataset()
#ds = ds.map(lambda inp: ( inp["transmitter_id"], inp["transmitter_id"])).cache().repeat(100000).batch(1000).prefetch(2000)
ds = ds.map(lambda inp: (inp["time_domain_IQ"], inp["transmitter_id"]))
ds = ds.take(2)
print(ds.element_spec)


for e in ds:
    print(e)

sys.exit(1)

inputs  = keras.Input(shape=(1,))
outputs = keras.layers.Dense(1)(inputs)

model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())
              #metrics=['accuracy'])

# model.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss=keras.losses.CategoricalCrossentropy(from_logits=False)
# )

model.fit(x=ds, epochs=10)
derp = tf.constant([[1,],])
print(derp)
print( model(derp))