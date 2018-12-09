'''
Here is the script to test the model on a image 

this code is in the end of the file main.py 
'''
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 

#read inpt image 

img = cv2.imread('input_test_image/nmber.jpg')
gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rescaled_image = cv2.resize(gray_image, (28,28))

plt.imshow(rescaled_image, cmap ='gray')
plt.show()

rescaled_image.shape

#normalization 
dum = rescaled_image.reshape(1,-1)/255  
dum.shape

# apply the trained tf model 
with tf.Session() as sess:
	#load the trained tf model
	saver.restore(sess, "./output_trained_model/tfmodel.ckpt")

	Z = output_layer.eval(feed_dict = {X:dum, keep_prob:1.0})
	y_pred = np.argmax(Z, axis = 1)

	print("Prediction for test image is {0}".format(y_pred))
