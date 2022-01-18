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

# +
x_train = [1, 2, 3, 4, 5]
y_train = [2, 3, 4, 5, 6]

# shape 1 (rank 1)
W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')
# H
hypothesis = x_train * W + b

# -

# cost(W, b)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
print(cost)

# +
#optimizaer = tf.train.GradientDescentOptimizer(learning_late=0.01)
sgd = tf.keras.optimizers.SGD(learning_rate=0.005)

model = tf.keras.models.Sequential() #모델
model.add(tf.keras.layers.Dense(1, input_dim = 1)) # 레이어 추가 
model.compile(loss='mean_squared_error', optimizer = sgd)

model.fit(x_train, y_train, epochs=1500, verbose=1) #verbose 0:로그x 1:로그o 2:간결로그o

print(model.predict([1]))

