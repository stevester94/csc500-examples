#! /usr/bin/python3


from steves_utils import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

vdsa = datasetaccessor.VanillaDatasetAccessor(
    #day_to_get=[2],
    #transmitter_id_to_get=[10],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")

ds = vdsa.get_dataset()
ds = ds.map(lambda inp: ([inp["transmission_id"],], [inp["transmission_id"],])) # This will just return tuples

for e in ds:
    print(e)
