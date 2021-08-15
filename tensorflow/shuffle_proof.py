#! /usr/bin/python3

# This example shows that our dataset does shuffle between iterations, and that it is deterministically shuffled
# across runs (As long as you set the seed)

from steves_utils import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

# Comment this out to see that the indices change between runs
tf.random.set_seed(1337)

vdsa = datasetaccessor.VanillaDatasetAccessor(
    day_to_get=[2],
    transmitter_id_to_get=[10],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")

ds = vdsa.get_dataset()
ds = ds.map(lambda inp: ([inp["transmission_id"],], [inp["transmission_id"],])) 

print("Iteration 1:")
for e in ds:
    print(e)


print("Iteration 2:")
for e in ds:
    print(e)
