#! /usr/bin/python3

# Toy example where we aren't even using a one hot encoding.


from steves_utils import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys

tf.random.set_seed(1337)


iq = tf.ones((2,128))
time_series = tf.ones((1,128))
image = tf.ones((3,10,10))
deep_image = tf.ones((20,10,10))

# iq_ds = tf.data.Dataset.from_tensors(iq).repeat(1000)
# image_ds = tf.data.Dataset.from_tensors(image).repeat(1000)


#####################################################################
inputs = keras.Input(shape=iq.shape)
net = keras.layers.MaxPooling1D(
    pool_size=7,
    strides=1,
    data_format="channels_first",
)(inputs)

print("{} input: {}, output: {}".format("IQ MaxPool1D", inputs.shape, net.shape))


#####################################################################
# inputs = keras.Input(shape=image.shape)
# net = keras.layers.MaxPooling1D(
#     pool_size=7,
#     strides=1,
#     data_format="channels_first",
# )(inputs)

# print("{} input: {}, output: {}".format("image MaxPool1D", inputs.shape, net.shape))

#####################################################################
inputs = keras.Input(shape=deep_image.shape)
net = keras.layers.MaxPooling2D(
    pool_size=7,
    strides=1,
    data_format="channels_first",
)(inputs)

print("{} input: {}, output: {}".format("image MaxPool2D", inputs.shape, net.shape))


#####################################################################
# inputs = keras.Input(shape=deep_image.shape)
# net = keras.layers.MaxPooling1D(
#     pool_size=7,
#     strides=1,
#     data_format="channels_first",
# )(inputs)

# print("{} input: {}, output: {}".format("deep_image MaxPool1D", inputs.shape, net.shape))

#####################################################################
inputs = keras.Input(shape=deep_image.shape)
net = keras.layers.MaxPooling2D(
    pool_size=7,
    strides=1,
    data_format="channels_first",
)(inputs)

print("{} input: {}, output: {}".format("deep_image MaxPool2D", inputs.shape, net.shape))

#####################################################################
inputs = keras.Input(shape=time_series.shape)
net = keras.layers.MaxPooling1D(
    pool_size=7,
    strides=1,
    data_format="channels_first",
)(inputs)

print("{} input: {}, output: {}".format("time_series MaxPool1D", inputs.shape, net.shape))

