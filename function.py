import numpy as np
import theano.tensor as T
from theano import function
import theano

# basic
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
# input param: [x,y]; output param: z
f = function([x, y], z)

print(f(2, 3))

# to pretty-print the function
from theano import pp

print(pp(z))

# matrix
x = T.dmatrices('x')
y = T.dmatrices('y')
z = x + y
f = function([x, y], z)

print(f(np.arange(12).reshape(3, 4), 10 * np.ones((3, 4))))

# activation function example
x = T.dmatrices('x')
s = 1 / (1 + T.exp(-x))  # np.exp() logistic or soft step
logistic = theano.function([x], s)
print(logistic([[0, 1], [-2, -3]]))

# multiply outputs for a function
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff__squared = diff ** 2
# 多返回值
f = theano.function([a, b], [diff, abs_diff, diff__squared])
x1, x2, x3 = f(
    np.ones((2, 2)),
    np.arange(4).reshape((2, 2)))
print(x1, x2, x3)

# name for a function
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
# 设置默认值: y = 1;设置默认的名字: w = 2,name = 'weights'
f = theano.function([x,
                     theano.In(y, value=1),
                     theano.In(w, value=2, name='weights')],
                    z)
print(f(23, ))
print(f(23, 2))
print(f(23, 2, weights=4))


