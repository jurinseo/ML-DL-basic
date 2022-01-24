# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# +
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]
# -

w1 = tf.Variable(tf.random.normal([1]), name='weight1')
w2 = tf.Variable(tf.random.normal([1]), name='weight2')
w3 = tf.Variable(tf.random.normal([1]), name='weight3')
b = tf.Variable(tf.random.normal([1]), name='bias')


learning_rate = tf.Variable(0.00001)

for i in range(2500):
    with tf.GradientTape(persistent=False) as tape:
        hypothesis = x1_data*w1 + x2_data * w2 + x3_data*w3 + b
#         hypothesis = tf.multiply(x1_data, w1) + tf.multiply(x2_data, w2) + tf.multiply(x3_data, w3) + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
        w1.assign_sub(learning_rate * w1_grad) 
        w2.assign_sub(learning_rate * w2_grad) 
        w3.assign_sub(learning_rate * w3_grad) 
        b.assign_sub(learning_rate * b_grad)   
        
    if i % 50 == 0:
        print("epochs: {:5} |cost:{:10.6f} |W1:{:10.4f} |W2:{:10.4f} |W3:{:10.4f} |b{:10.6f}".format(i, cost.numpy(), w1.numpy()[0], w2.numpy()[0], w3.numpy()[0], b.numpy()[0]))

# Matrix
x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# +
W = tf.Variable(tf.random.normal([1, 5]))
b = tf.Variable(tf.random.normal([1]), name='bias')

learning_rate = tf.Variable(0.00001)
optimizer = tf.keras.optimizers.SGD(learning_rate)

# -

for i in range(2500):
    with tf.GradientTape(persistent=False) as tape:
        hypothesis = tf.matmul(W, x_data) + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        grads = tape.gradient(cost, [W, b])
        
        optimizer.apply_gradients(zip(grads, [W, b]))
    if i % 50 == 0:
        print("epochs: {:5} |cost: {:10.6} |W1: {:10.4} |W2: {:10.4} |W3: {:10.4}| b: {:10.6}"
              .format(i, cost, W.numpy()[0][0], W.numpy()[0][1], W.numpy()[0][2], b.numpy()[0]))





