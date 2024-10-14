import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_data():

    return x,y 


min_val = 0.0
max_val = 2.0
num_steps = 50
step_size = (max_val - min_val)/(num_steps - 1)
m = tf.Variable(tf.zeros(shape=[num_steps]))
loss = tf.Variable(tf.zeros(shape=[num_steps]))
for i in range(0,num_steps):
    m[i].assign(min_val + i * step_size)
    e = y - m[i] * x 
    loss[i].assign(tf.reduce_sum(tf.multiply(e,e))/len(x))
i = tf.argmin(loss)
m_best = m[i].numpy()


def plot_linear_model():

    
