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
tf.random.set_seed(100)

xy = np.loadtxt('Data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape)

# +
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

W = tf.Variable(tf.random.normal([x_data.shape[1],1]), name='weight') 
b = tf.Variable(tf.zeros([1]), name='bias')  


# -

def logistic_regression(X):
    h = tf.divide(1., 1. + tf.exp(tf.matmul(X, W) + b))
    return h


# +
def loss_function(h, X, Y):
    cost = tf.reduce_mean(-Y * tf.math.log(h) - (1 - Y)*tf.math.log(1 - h))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05)


# -

def accuracy_function(h, Y):
    predicted = tf.cast(h > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    return accuracy


def grad(h, X, Y):
    with tf.GradientTape() as tape:
        loss_value = loss_function(logistic_regression(X), X, Y)
        return tape.gradient(loss_value, [W, b])


epochs = 2500
i = []
cost = []
acc = []
for epoch in range(epochs):
    i.append(epoch)
    for X, Y in iter(dataset):
        grads = grad(logistic_regression(X), X, Y)
        optimizer.apply_gradients(grads_and_vars = zip(grads, [W, b]))
        cost.append(loss_function(logistic_regression(X), X, Y))
        acc.append(accuracy_function(logistic_regression(X), Y))
        if epoch % 100 == 0:
            print("Iter: {}, Loss: {:.5f}, Accuracy: {:.4f}".format(epoch, loss_function(logistic_regression(X), X, Y), accuracy_function(logistic_regression(X), Y)))

plt.rcParams["figure.figsize"] = (8,6)
plt.plot(i, cost, "b")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()

plt.rcParams["figure.figsize"] = (8,6)
plt.plot(i, acc, "r")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


