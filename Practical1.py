#importing Tensorflow and printing out the version
import tensorflow as tf
print(tf.__version__)
print()
#creating scaler using tf.constant
print()
scalar=tf.constant(7)
print(scalar)
print(scalar.ndim)
print()
#create vector using tf.constant
print()
vector=tf.constant([10,10])
print(vector)
print(vector.ndim)
print()
#creating matrix\
print()
matrix=tf.constant([[10,11],[12,13]])
print()
print(matrix)
print(matrix.ndim)
#tensorflow having default data type is int32
#simple calculation usig tensor
basic_tensor=tf.constant([[10,11],[12,13]])
print()
print(basic_tensor)
print()
#we can add,subtract,multiply anddivide valuein atensor using the basic operator
print(basic_tensor+10)
print()
print(basic_tensor-10)
print()
print(basic_tensor*10)
print()
print(basic_tensor/10)
print(basic_tensor%10)

