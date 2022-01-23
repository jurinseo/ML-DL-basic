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

# +
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis for linear model X * W, cost function
def cost_function(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))

W_values = 0.1 * np.linspace(-30, 50, num = 81)
cost_values = []

for i in np.linspace(-30, 50, num = 81):
    feed_W = i * 0.1
    current_cost = cost_function(feed_W, X, Y)
    cost_values.append(current_cost)
    print("W: {:.3f} | cost: {:.3f}".format(feed_W, current_cost))


plt.rcParams["figure.figsize"] = (8,6)
plt.plot(W_values, cost_values, "b")
plt.xlabel("W")
plt.ylabel("Cost(W)")
plt.show()




# +
# Gradient Descent MANUAL

tf.random.set_seed(1) # seed 설정

#W = tf.Variable(tf.random.normal([1]))
W = tf.Variable([-1.1600207])
print(W)
W_values = []
cost_values = []
for step in range(500) :
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    W_values.append(W.numpy())
    cost_values.append(cost.numpy())
    learning_rate = 0.001
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, 2*X))
    descent = tf.subtract(W, tf.multiply(learning_rate, gradient))
    #print(descent)
    #print(W)
    
    W.assign(descent)
    print("step : {:5} | cost : {:5.6f} | W : {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))
    
    
    
    #if step % 10 == 0:
    #    print("step : {:5} | cost : {:5.6f} | W : {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))
#print(W_values)
#print(cost)




# +
# # +
plt.rcParams["figure.figsize"] = (8,6)
plt.plot(W_values, cost_values, "o")
plt.xlabel("W")
plt.ylabel("Cost(W)")
plt.show()

plt.rcParams["figure.figsize"] = (8,6)
plt.plot(np.linspace(1, 500, num = 500), cost_values, "g")
plt.xlabel("epoch")
plt.ylabel("Cost(W)")
plt.show()

# # +


# + endofcell="--"
# Gradient Descent COMPUTE
from tensorflow.keras.utils import plot_model
tf.random.set_seed(1) # seed 설정

#W = tf.Variable(tf.random.normal([1]))

sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
model = tf.keras.models.Sequential() #모델
model.add(tf.keras.layers.Dense(1, input_dim = 1, use_bias=False)) # 레이어 추가
model.compile(loss='mean_squared_error', optimizer = sgd)
for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  #print("Bias")
  #print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')
history = model.fit(X, Y, epochs=500, verbose=1) #verbose 0:로그x 1:로그o 2:간결로그o


# -



# --

plt.rcParams["figure.figsize"] = (8,6)
plt.plot(history.history['loss'])

# +
# simple manul model by JS, Choi
X_ = tf.constant(X,dtype=tf.float32)
my_val = [-1.1600207]
W = tf.Variable(my_val, name = 'weight')
alpha = 0.001

@tf.function
def tftrain():
    with tf.GradientTape(persistent=True) as tape:
        hypothesis = tf.multiply(X_,W)
        activate = hypothesis - Y
        cost = tf.reduce_mean(tf.square(activate))
        dw = tape.gradient(cost,W)
        W.assign(W - alpha*(dw))

        
for n in range(500):
    tftrain()
   

# -



