#! /usr/bin/python3
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

# The below is basically how our data is represented
# Each one of the below two_d_* tensors can be considered as the contents of a single binary data file.


# These have shape 2,6 [row][col]
# So, it's a little wonky. The indexings is [I or Q][Sample Index]
two_d_A = tf.constant(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 1, 1, 2]
    ]
)

two_d_B = tf.constant(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 1, 1, 2]
    ]
)

two_d_C = tf.constant(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 1, 1, 2]
    ]
)

# Note that 'from_tensors' creates a dataset with just one element. In this case it's a single tensor of (3, 2, 6).
ds = tf.data.Dataset.from_tensors([two_d_A, two_d_B, two_d_C])
print("from_tensors Element spec: ", ds.element_spec) # (3,2,6)
print("from_tensors cardinality: ", ds.cardinality()) # 1


# From tensor slices is what you really want. What it's doing under the hood is slicing apart the first dimension.
# In other words, it's assuming the input is a list of tensors, and splits it apart at the first dimension.
ds = tf.data.Dataset.from_tensor_slices([two_d_A, two_d_B, two_d_C])
print("from_tensors Element spec: ", ds.element_spec) # (2,6)
print("from_tensors cardinality: ", ds.cardinality()) # 3