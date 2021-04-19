#! /usr/bin/python3
import sys

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import datasetaccessor


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
#frames = tf.signal.frame(two_d_A, multiple, multiple)
#frames = tf.transpose(frames, perm=[1,0,2])
#print("frames: ", frames)
# Now we can stick that into a dataset
#ds = tf.data.Dataset.from_tensor_slices(frames) # This ds has 
#print("signal ds cardinality: ", ds.cardinality()) # 3 elements
#print("signal ds element_spec: ", ds.element_spec) # shape (2,48)



##########################################################
# Realistic Example
##########################################################
vdsa = datasetaccessor.VanillaDatasetAccessor(
    day_to_get=[3],
    transmitter_id_to_get=[10],
    tfrecords_path="../csc500-dataset-preprocessor/vanilla_tfrecords/")

origin = vdsa.get_dataset()

# for e in ds:
#     print("yue")

###########################################################
# All of the below are day 2, transmitter_id 10, unless otherwise noted
###########################################################

# 2m14.442s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]).map(lambda y: (y, x["transmitter_id"])),
# )

# 0m2.247s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })

# 2m22.092s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]).map(lambda y: (y, x["transmitter_id"])),
    
# )
# ds = ds.prefetch(1000)

# 2m24.955s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]).map(lambda y: (y, x["transmitter_id"])),
#     num_parallel_calls=tf.data.AUTOTUNE,
#     deterministic=True
# )
# ds = ds.prefetch(1000)

# 1m27.797s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]),
#     num_parallel_calls=tf.data.AUTOTUNE,
#     deterministic=True
# )
# ds = ds.prefetch(1000)

# 1m33.143s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]),
#     num_parallel_calls=10,
#     deterministic=False
# )
# ds = ds.prefetch(1000)

# Day 3, transmitter 10
# There's an initial balls to the wall load from disk, but then we are totally constrained by what looks like to be one thread.
# 1m31.147s
# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]),
#     num_parallel_calls=tf.data.AUTOTUNE,
#     deterministic=True
# )
# ds = ds.prefetch(1000)


# ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
# ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })
# def batch_generator(t, batch_size):
#     # Split it on the first dimension, batch_size number of times, the remainder gets dropped
#     num_full_batches = int(t.shape[0] / batch_size)
#     remainder = t.shape[0] % batch_size

#     splits = [batch_size] * num_full_batches
#     splits.append(remainder)

#     # Drop the remainder
#     for split_tensor in tf.split(t, splits)[:-1]:
#         yield split_tensor
    
#     raise StopIteration

# ds = ds.interleave(
#     lambda x: tf.data.Dataset.from_tensor_slices(x["time_domain_IQ_frames"]),
#     num_parallel_calls=tf.data.AUTOTUNE
# )


ds = origin.map( lambda x: (tf.signal.frame(x['time_domain_IQ'], 48, 48), x["transmitter_id"] ))
ds = ds.map( lambda time_domain_IQ, transmitter_id: {'time_domain_IQ_frames': tf.transpose(time_domain_IQ, perm=[1,0,2]), 'transmitter_id': transmitter_id  })


print("ds cardinality: ", ds.cardinality())
print("ds element_spec: ", ds.element_spec)

counter = 0
for e in ds:
    counter += 1

print(counter)




























sys.exit(0)




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