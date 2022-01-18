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

# +
hello = tf.constant("Hello, TensorFlow!")

print(hello)
# -

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1:{}, node2:{}".format(node1, node2))
print("node3:{}".format(node3))


# FUNCTION
@tf.function
def add(node1, node2):
    return node1 + node2


node3 = add(node1, node2)
print(node3)

node4 = tf.constant([[1, 2, 3], [4, 5, 6]])
node5 = tf.constant([[2, 2, 2], [3, 3, 3]])
node6 = add(node4, node5)
print(node6)
