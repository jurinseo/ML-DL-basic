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
import numpy as np
tf.random.set_seed(777)

x_train = [[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2.]]
y_train = [[0.],[0.],[0.],[0.],[0.],[1.]]

# +
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.zeros([2,1]), name='weight') 
b = tf.Variable(tf.zeros([1]), name='bias')    


# -

def logistic_regression(X):
    h = tf.divide(1., 1. + tf.exp(tf.matmul(X, W) + b))
    return h


# +
def loss_function(h, X, Y):
    cost = tf.reduce_mean(-Y * tf.math.log(h) - (1 - Y)*tf.math.log(1 - h))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)


# -

def grad(h, X, Y):
    with tf.GradientTape() as tape:
        loss_value = loss_function(logistic_regression(X), X, Y)
        return tape.gradient(loss_value, [W, b])


# +
epochs = 2501

for epoch in range(epochs):
    for X, Y in iter(dataset):
        grads = grad(logistic_regression(X), X, Y)
        optimizer.apply_gradients(grads_and_vars = zip(grads, [W, b]))
        
        if epoch % 50 == 0:
            print("Iter: {}, Loss: {:.4f}".format(epoch, loss_function(logistic_regression(X), X, Y)))
# -




