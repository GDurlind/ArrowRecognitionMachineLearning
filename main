#!/usr/bin/env python3

# Load required packages
#import scipy.io as sio
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
from tensorflow.keras import models
#from tensorflow.keras import layers
#from tensorflow.keras.layers import Dense, Dropout, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.cm as cm
#import numpy.ma as ma
from PIL import Image                                                            
import glob
import os
import random
import rospy
from geometry_msgs.msg import Twist


move = Twist()

# Add here the required packages to connect to ROS 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# List of arrow classes
namesList = ['up', 'down', 'right', 'left']

# folder names of testing images
imageFolderPath = r'/u/u/ghd25/Exercise3/Database_arrows'
imageFolderTestingPath = imageFolderPath + r'/validation'
imageTestingPath = []

# full path to testing images
for i in range(len(namesList)):
    testingLoad = imageFolderTestingPath + '/' + namesList[i] + '/*.jpg'
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)
    
# print number of images for testing
print(len(imageTestingPath))

# resize images to speed up training process
updateImageSize = [128, 128]
tempImg = Image.open(imageTestingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

# create space to load testing images
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

# load testing images
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')

# create space to load testing labels
testLabels = np.zeros((len(x_test),))
  
# load testing labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        testLabels[countPos,] = i
        countPos = countPos + 1

# convert testing labels to one-hot format
testLabels = tf.keras.utils.to_categorical(testLabels, len(namesList))
 
# Load pre-trained model
CNN_model = models.load_model(r'/rosdata/robot_ws/src/pub_sub/src/CNN_Model.h5')

    
# Add here the code to prepare you script to work with ROS
if __name__ == '__main__':
    rospy.init_node("arrow_recognition")
    rospy.loginfo("Node has been started")
    rate = rospy.Rate(0.1)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    # Loop to read input digits
    while not rospy.is_shutdown():
        
        # write here the required code to do the following:
        # select an arrow image randmonly from the test dataset
        randomImage = random.randint(0, x_test.shape[0]+1)
        testImage = x_test[randomImage,:,:]
        plt.imshow(testImage.squeeze())
        plt.show()
        
        # select the corresponding class label
        testLabel = testLabels[randomImage]
        
        testImage = testImage.reshape((1, 96, 128, 1))
    
        batchPredict = CNN_model.predict_on_batch(testImage)
        
        print('Real digit: ', testLabel, '    Predicted digit:', np.argmax(batchPredict, axis=1))
        maxPredict = (np.argmax(batchPredict, axis=1))
        if (maxPredict == 0):
            print('Robot will move forward')
        if (maxPredict == 1):
            print('Robot will move backward')
        if (maxPredict == 2):
            print('Robot will turn right')
        if (maxPredict == 3):
            print('Robot will turn left')
            
        input('Press enter to start movement:')
        print('Robot moving')
        

        
        if (maxPredict == 0):
            move.linear.x = 0.2
            move.angular.z = 0
            
        if (maxPredict == 1):
            move.linear.x = -0.2
            move.angular.z = 0
            
        if (maxPredict == 2):
            move.linear.x = 0.2
            move.angular.z = 0.6         
            
        if (maxPredict == 3):
            move.linear.x = 0.2
            move.angular.z = -0.6
            
        pub.publish(move)
        
        input('Press enter to load next arrow:')
         
        #plot
        
        # prepare the image with the correct shape for the CNN
        # use the input image for prediction with the pre-trained model
        # use the predicted output to control a mobile robot in CoppeliaSim via ROS
        # show int the terminal (or plots) the actual and predicted arrow
        # repeate the process until stopped by the user
        pass
    
    rospy.spin()
print('OK')
