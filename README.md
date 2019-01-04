# tf_notes
tf notes

```

TensorFlow Notes

Alvaro Fuentes. “Mastering Predictive Analytics with scikit-learn and TensorFlow.” iBooks. 

- API

Low level: TensorFlow Core, flexibility
High level: tf.contrib.learn/keras/TF-Slim, fast implementation

- First tensorflow program

import tensorflow as tf
hello=tf.constant("Hello") # create a constant named hello
sess=tf.Session() # create a session
print(sess.run(hello)) # print hello

- Tensors

[2., 2., 1.] a vector of shape 3
[[9., 5., 3.], [4., 5., 7]] a matrix with shape [2, 3]
[[[8., 3.]], [[7., 9.,]]] a matrix with shape [2, 1, 2]

- Session:
an object that encapsulates the environment in which operation objects are executed.
So, sessions are objects that place operations onto devices such as CPUs or GPUs.

- Placeholders:
A placeholder is a promise to provide a value later.
These objects are usually used to provide training and testing values in machine learning models.

- Variables:
These are objects that are initialized with a value, 
and that value can change during the execution of the graph.
Typically, they are used as trainable variables in machine learning models.

- Constants:
Constants are objects whose values never change

- Computational graph

f(x,y)=(x*x)*y+4*y

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
c = tf.constant(5)

square_node = x*x
mult_node = square_node*y
quadruple_node = 4*y
adder_node = mult_node + quadruple_node

sess = tf.Session()
sess.run(c)
sess.run(x,feed_dict={x:6})
sess.run(square_node,feed_dict={x:10})
sess.run(adder_node,feed_dict={x:3,y:2})

- equivalently

f=x**2*y+4*y
sess.run(f,feed_dict={x:3,y:2})

- two ways of running objects

run() # from a session
eval() # from the tensor

with tf.Session() as sess:
  print("f(10,5)=",sess.run(f,feed_dict={x:10,y:5}))
  print("f(10,5)=",f.eval(feed_dict={x:10,y:5}))

- with statement

with expression [as variable]:
    with-block

1. expression is evaluated
2. expression results in an object that supports __enter__() and __exit__()
3. The object's __enter__() is called before with-block
4. execution of the with-block
5. The object's __exit__() method is called

- Linear Model

y=b+wx+noise

# generate data

np.random.seed(123)
x_train=np.arrange(0,10,0.25)
y_train=5*x_train+1+np.random.normal(0,1,size=x_train.shape)
plt.scatter(x_train,y_train)

# trainable parameters

w = tf.Variable(0.0,dtype=tf.float32) 
b = tf.Variable(0.0,dtype=tf.float32)

# placeholders for data

x = tf.placeholder(tf.float32) # a numeric value not vector
y = tf.placeholder(tf.float32) # a numeric value not vector

# model

yhat = w*x+b

# loss function (calculated as sum squared errors)

loss=tf.reduce_sum(tf.square(yhat-y))

# Use gradient descent to optimize

my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0005)
training_op=my_optimizer.minimize(loss)

# initial values

init = tf.global_variables_initializer()
sess = tf.Session() # create a session
sess.run(init) # after this step the variables in the session have been initialized

# Train

for i in range(20):
  sess.run(training_op,feed_dict={x:x_train,y:y_train})
  print("Iteration {}: w: {:0.5f}, b: "{:0.5f}".format(i,sess.run(w),sess.run(b))
```
