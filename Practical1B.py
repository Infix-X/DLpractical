from ctypes import cast
import theano
import theano.tensor as T
# Change this line
convert_to_bool = cast(bool, name='convert_to_bool')


# Declaring variables
a = T.dscalar()
b = T.dscalar()

# Subtracting
res = a - b

# Creating a function
sub = theano.function([a, b], res)

# Test the function
print(sub(1.5, 2.0))  # Example values, output will be -0.5


