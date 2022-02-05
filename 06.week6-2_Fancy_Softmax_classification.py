# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
tf.random.set_seed(1)

xy = np.loadtxt('Data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7
#분류를 위해 라벨값을 one-hot encoding
Y_one_hot = tf.one_hot(y_data.astype(np.int32), nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print(Y_one_hot)


print(x_data.shape, Y_one_hot.shape)

W = tf.Variable(tf.random.normal((x_data.shape[1], nb_classes)), name ='weight')
b = tf.Variable(tf.random.normal((nb_classes, )), name = 'bias')
variables = [W, b]


# +
def logit_fn(X):
    return tf.matmul(X, W) + b


def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))


def cost_fn(X, Y):
    logits = logit_fn(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient([Y])))
    return cost



def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
        return grads

    
def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_pred = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy



# +
def fit(X, Y, epochs=1001, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05)
    
    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(grads_and_vars = zip(grads, variables))
        if (i == 0) | ((i+1) % 100 == 0):
            print('epochs: {} |Loss: {}| Accuracy: {}'.format(i+1, cost_fn(X, Y).numpy(), prediction(X, Y).numpy()))
            

# -

fit(x_data, Y_one_hot)

sample = [[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 4, 0, 0, 1]]
sample = np.asarray(sample, dtype = np.float32)

a = hypothesis(sample)
print(a)


