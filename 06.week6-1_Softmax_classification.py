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

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# +
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

nb_classes = 3

# +
W = tf.Variable(tf.random.normal([x_data.shape[1], nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

print(dataset)


# +
def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W)+b)

print(hypothesis(x_data))

# +
sample_db = [[8,2,1,4]]
sample_db = np.asarray(sample_db, dtype=np.float32)

print(hypothesis(sample_db))


# +
def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logits), axis=1)
    cost_mean = tf.reduce_mean(cost)
    
    return cost_mean

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
print(cost_fn(x_data, y_data))


# +
def grad(X, Y):
    with tf.GradientTape() as tape:
        loss_value = cost_fn(X, Y)
        return tape.gradient(loss_value, [W, b])
    
print(grad(x_data, y_data))
# -

epoch = []
cost = []
def fit(X, Y, epochs = 3001):
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
    
    for i in range(epochs):
        grads = grad(X, Y)
        optimizer.apply_gradients(zip(grads, [W, b]))
        epoch.append(i)
        cost.append(cost_fn(X,Y).numpy())
        if i % 100 == 0:
            print('epoch: {}, cost: {:.4f}'.format(i, cost_fn(X, Y).numpy()))


fit(x_data, y_data)

plt.rcParams["figure.figsize"] = (8,6)
plt.plot(epoch, cost, "b")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()

sample = [[1, 7, 7, 7]]
sample = np.asarray(sample, dtype = np.float32)

a = hypothesis(sample)
print(a)


