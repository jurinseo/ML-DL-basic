# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

import numpy as np
import tensorflow as tf

# +
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
# +
W = tf.Variable(tf.random.normal([3, 1], name='weight'))
b = tf.Variable(tf.random.normal([1,], name='bias'))

learning_rate = tf.Variable(0.00002)
optimizer = tf.keras.optimizers.SGD(learning_rate, momentum = 0.3)

def predict(X):
    return tf.matmul(X, W) + b


# -


for i in range(2500):
    with tf.GradientTape(persistent=False) as tape:
        hypothesis = tf.matmul(x_data, W) + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        W_grad, b_grad= tape.gradient(cost, [W, b])
#         optimizer.apply_gradients(zip(grads, [W, b]))
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)
        
    if i % 50 == 0:
        print("epochs: {:5} |cost: {:10.6} |W1: {:10.4} |W2: {:10.4} |W3: {:10.4} | b: {:10.6}"
              .format(i, cost, W.numpy()[0][0], W.numpy()[1][0], W.numpy()[2][0], b.numpy()[0]))
        

print(W.numpy())
print(x_data)
print(y_data)

predict(x_data).numpy()

predict([[89., 95., 92], [11, 35, 85]]).numpy()


