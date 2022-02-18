# 00. TensorFlow Fundamentals Exercises
# https://github.com/mrdbourke/tensorflow-deep-learning#-00-tensorflow-fundamentals-exercises

# get libraries
import tensorflow as tf
import numpy as np

# 1. Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
scalar = tf.constant(0)
vector = tf.constant([0, 0])
matrix = tf.constant(
    [[0, 0],
     [0, 0],
     [0, 0]]
)
tensor = tf.constant(
    [[[0, 0, 0, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0]]]
)

print(f"""
========== Q1 ==========
scalar:\n{scalar}\n
vector:\n{vector}\n
matrix:\n{matrix}\n
tensor:\n{tensor}
""")


# 2. Find the shape, rank and size of the tensors you created in 1.
print(f"""
========== Q2 ==========
shape, rank, and size of scalar: {scalar.shape, scalar.ndim, tf.size(scalar)}
shape, rank, and size of vector: {vector.shape, vector.ndim, tf.size(vector)}
shape, rank, and size of matrix: {matrix.shape, matrix.ndim, tf.size(matrix)}
shape, rank, and size of tensor: {tensor.shape, tensor.ndim, tf.size(tensor)}
""")


# 3. Create two tensors containing random values between 0 and 1 with shape [5, 300].
random_tensor_shape = [5, 300]
random_tensor_1 = tf.random.uniform(
    shape=random_tensor_shape,
    minval=0,
    maxval=1,
    dtype=tf.float32,
    seed=420
)
random_tensor_2 = tf.random.uniform(
    shape=random_tensor_shape,
    minval=0,
    maxval=1,
    dtype=tf.float32,
    seed=69
)

print(f"""
========== Q3 ==========
random tensor #1: {random_tensor_1}
random tensor #2: {random_tensor_2}
""")


# 4.

# 5.

# 6.

# 7.

# 8.

# 9.

# 10.
