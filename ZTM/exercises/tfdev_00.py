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
A = tf.random.uniform(
    shape=[5, 300],
    minval=0,
    maxval=1,
    dtype=tf.float32,
    seed=420
)
B = tf.random.uniform(
    shape=[5, 300],
    minval=0,
    maxval=1,
    dtype=tf.float32,
    seed=69
)

print(f"""
========== Q3 ==========
random tensor #1: {A}
random tensor #2: {B}
""")


# 4. Multiply the two tensors you created in 3 using matrix multiplication.
print(f"""
========== Q4 ==========
{A @ tf.transpose(B)}
""")


# 5. Multiply the two tensors you created in 3 using dot product.
print(f"""
========== Q5 ==========
{tf.tensordot(A, tf.transpose(B), axes=1)}
""")


# 6. Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
C = tf.random.normal([224, 224, 3])

print(f"""
========== Q6 ==========
{C}
""")

# 7. Find the min and max values of the tensor you created in 6 along the first axis.
C_min_1axis = tf.math.reduce_min(C)
C_max_1axis = tf.math.reduce_max(C)

print(f"""
========== Q7 ==========
min value of first axis: {C_min_1axis}\n
max value of first axis: {C_max_1axis}\n
""")

# 8. Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
D = tf.random.normal([1, 224, 224, 3])
D_squeezed = tf.squeeze(D)

print(f"""
========== Q8 ==========
original tensor shape: {D.shape}
squeezed tensor shape: {D_squeezed.shape}
""")

# 9. Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
E = tf.random.normal([10])

print(f"""
========== Q9 ==========
tensor's values: {E}
tensor's max value: {tf.reduce_max(E)}
tensor's max value's position: {tf.argmax(E)}
""")

# 10. One-hot encode the tensor you created in 9.
print(f"""
========== Q10 ==========
original tensor: {E}
one-hot encoded tensor: {tf.one_hot(
    indices=tf.cast(E, dtype=tf.int32).numpy(),
    depth=tf.size(E).numpy())}
""")
