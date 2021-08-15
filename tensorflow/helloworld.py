#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


from steves_utils import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys

tf.random.set_seed(1337)

vdsa = datasetaccessor.VanillaDatasetAccessor(
    day_to_get=[2],
    #transmitter_id_to_get=[1,2],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")

# We're just gonna train an identity function
# Fetching the data is slow because we are also grabbing all of the radio data. We cache is and repeat to make this fast AF

ds = vdsa.get_dataset()
ds = ds.map(lambda inp: ( inp["transmitter_id"], inp["transmitter_id"])).cache().repeat(100000).batch(1000).prefetch(2000)
#ds = ds.map(lambda inp: (inp["time_domain_IQ"], inp["transmitter_id"]))

#ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
#ds = ds.map(lambda inp: ( inp, inp )).repeat(100000).batch(1000).prefetch(2000)


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
