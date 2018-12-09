'''
Please see the readme file 

Here is the main code 

Author : GHNAMI Helmi 
'''

# import the required lib and package 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 

# import the MNIST dataset 
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets("./data", one_hot = True)

#Define the neural network parameter
n_input = 784   # input image size of 28 x28
n_hidden_1 =512  # First hidden layer with 512 neuron 
n_hidden_2 =256  # Second hidden layer with 256 neuron
n_hidden_3 =128  # Third hidden layer with 128 neuron 
n_output = 10    #output layer with 10 class (0-9 digit ) 

# Define thehyperparameters fo the training using tensorflow 

learning_rate = 1e-4
epochs = 3000
batch_size = 128 
keep_prob = tf.placeholder(tf.float32)

""" Remark :For instance, let's say you have 1050 training samples and you want to set up a batch_size equal to 100.
 The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network.
 Next it takes the second 100 samples (from 101st to 200th) and trains the network again. 
 We can keep doing this procedure until we have propagated through all samples of the network. 
 A problem usually happens with the last set of samples. In our example we've used 1050 which is not divisible by 100 without remainder. The simplest 
solution is just to get the final 50 samples and train the network.
"""

#  Building the tensorflow graph 
X = tf.placeholder(tf.float32,[None, n_input])
Y = tf.placeholder(tf.float32,[None, n_output])

#weight definition 
nn_weight = {"W1":tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev = 0.1)),
			 "W2":tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev = 0.1)),
			 "W3":tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev = 0.1))
			 "Wout":tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev = 0.1))
			 }
# bias definition 
nn_bias =   {"B1":tf.Variable(tf.truncated_normal([n_hidden_1])),
			 "B2":tf.Variable(tf.truncated_normal([n_hidden_2])),
			 "B3":tf.Variable(tf.truncated_normal([n_hidden_3]))
			 "B4":tf.Variable(tf.truncated_normal([n_output]))
			 }

# Create the Neural Network model 

nn_layer_1 = tf.add(tf.matmul(X, nn_weight["W1"], nn_bias["B1"]))
nn_layer_2 = tf.add(tf.matmul(nn_layer_1, nn_weight["W2"], nn_bias["B2"]))
nn_layer_3 = tf.add(tf.matmul(nn_layer_2, nn_weight["W3"], nn_bias["B3"]))
layer_drop = tf.nn.dropout(nn_layer_3, keep_prob)
output_layer = tf.add(tf.matmul(layer_drop, nn_weight["Wout"], nn_bias["B4"]))


print('OK')



# run the algorithme
if __name__ == '__main__':
	print('Processing finished')
