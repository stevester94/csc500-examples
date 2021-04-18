#! /usr/bin/python3
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import sys


##########################################################
# Signal Example (this is the good shit)
##########################################################
multiple = 48
num_multiples=3
second_row_start = 1000
two_d_A = tf.constant(
    [
        list(range(multiple*num_multiples)),
        list(range(second_row_start, second_row_start + multiple*num_multiples))
    ]
)


# tf.siganl.frame works how you would hope tf.windowing works: it returns a single tensor in the form below:
# The results is a tensor of [channel][frame index][values], so we transpose in order to get the first index to be the frames
frames = tf.signal.frame(two_d_A, multiple, multiple)
frames = tf.transpose(frames, perm=[1,0,2])
print("frames: ", frames)
# Now we can stick that into a dataset
ds = tf.data.Dataset.from_tensor_slices(frames) # This ds has 
print("signal ds cardinality: ", ds.cardinality()) # 3 elements
print("signal ds element_spec: ", ds.element_spec) # shape (2,48)
sys.exit(1)


#############################
# Basic Example
#############################
print("Basic Example")

# So, you want to know why all this bullshit is required?
# Window returns a dataset of windows. The windows themselves are datasets (how fun!). So we have to use a mapping function to turn those datasets
# into tensors, HOWEVER flat_map will seriously just flatten the fuck out of everything, so we have to use a matching batch call
# in order to preserve the window-as-a-tensor structure.
basic_ds = tf.data.Dataset.range(240).window(48).flat_map(lambda window: window.batch(48))
print("basic_ds element_spec: ", basic_ds.element_spec)
for e in basic_ds:
    print(e)

# Results in 5, 48 element tensors. Marvelous!


#######################################################
# Realistic Example
#######################################################
print("Begin realistic example")

two_d_A = tf.constant(
    [
        list(range(96)),
        list(range(96))
    ]
)

ds = tf.data.Dataset.from_tensors(two_d_A) # This ds has 1 element, of 2,96
print("before element_spec: ", ds.element_spec)
print("before cardinality: ", ds.cardinality())

# interleave is the right way to go since its basically a map that expects datasets, and then concats those datasets
# ds = ds.interleave(
#     lambda x: 
#         tf.data.Dataset.zip(
#             (
#                 tf.data.Dataset.from_tensors(x[0]).window(48).flat_map(lambda window: window.batch(48)),
#                 tf.data.Dataset.from_tensors(x[1]).window(48).flat_map(lambda window: window.batch(48))
#             )
#         )
# )

# ds = ds.interleave(
#     lambda x: 
#                 tf.data.Dataset.from_tensors(x[0]).window(48).flat_map(lambda window: window.batch(48))
# )

print("after element_spec: ", ds.element_spec)


for e in ds:
    print(e)

sys.exit(0)
#######################################################
# Slightly More Complicated Example
#######################################################
print("Slightly More Complicated Example")
one_d_A = tf.constant([1,2,3,0])
one_d_B = tf.constant([4,5,6,0])
one_d_C = tf.constant([7,8,9,0])

print("one_d_A shape: ", one_d_A.shape) # (6,)


# Note we have 
ds = tf.data.Dataset.from_tensor_slices([one_d_A, one_d_B, one_d_C]) # This ds has 3 elements, tensors of (6,)
print("ds element_spec: ", ds.element_spec)
print("ds cardinality: ", ds.cardinality())

ds = ds.window(2, drop_remainder=True)
print("windowed ds element_spec: ", ds.element_spec)
for e in ds:
    print(list(e.as_numpy_iterator()))





sys.exit(0)
two_d = tf.constant(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 1, 1, 2]
    ]
)
print("two_d shape: ", two_d.shape) # (2,6)