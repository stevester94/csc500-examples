#! /usr/bin/python3
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

#inputs  = keras.Input(shape=(1,))
#outputs = keras.layers.Dense(1)(inputs)



zero_d = tf.constant(0)
print(zero_d) # Shape ()
print(zero_d.ndim) # 0


one_d = tf.constant([1, 2, 3, 4, 5, 6])

# Note that the trailing comma is just to denote that it's actually a tuple of one element
print(one_d) # Shape 6,

# Note that ndim is the number of indices needed to index this fucker
# ndim == order, degree, rank
print(one_d.ndim) # 1


# This has shape 2,6 [row][col]
two_d = tf.constant(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 1, 1, 2]
    ]
)
print(two_d) # shape=(2,6)
print(two_d.ndim) # 2

ds = tf.data.Dataset.from_tensors(one_d)
print("Element spec: ", ds.element_spec)  # This has a shape of (6,)

ds = tf.data.Dataset.from_tensors(one_d).batch(100)
print("Element spec: ", ds.element_spec) # This has a shape of (None, 6)

ds = tf.data.Dataset.from_tensors(zero_d).batch(100)
print("Element spec: ", ds.element_spec) # This has a shape of (None)



ds = tf.data.Dataset.from_tensors(two_d)
print("Element spec: ", ds.element_spec) # This has a shape of (2,6)

print("Elements:")
for e in ds:
    print(e)