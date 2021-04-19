#! /usr/bin/python3


import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

#vdsa = datasetaccessor.VanillaDatasetAccessor(tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")
vdsa = datasetaccessor.VanillaDatasetAccessor(
    day_to_get=[2],
    transmitter_id_to_get=[10],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")

# Shuffled: 0m12.887s
# Unshuffled: 0m15.7s
ds = vdsa.get_dataset()
ds = ds.map(lambda inp: ([inp["transmission_id"],], [inp["transmission_id"],])) # We just get tuples of these fuckers

print("Iteration 1:")
for e in ds:
    print(e)


print("Iteration 2:")
for e in ds:
    print(e)