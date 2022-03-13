# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:22:26 2022

@author: anjan
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU

inputs = tf.keras.Input(shape=(3,4,4))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
    

def conv_block(image_input, filters,max_pooling=True):
    conv= Conv2D(filters=filters,padding='valid',
    kernel_size=(3,3),kernel_initializer='he_normal',activation='relu')(image_input)
    
    conv=Conv2D(filters=filters,padding='valid',
    kernel_size=(3,3),kernel_initializer='he_normal',activation='relu')(conv)
    
    
    if max_pooling:
        next_layer = MaxPooling2D((2, 2))(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer,skip_connection




def unet_model(input_size,n_filters,n_classes):
    cblock1 = conv_block(inputs,n_filters)
    # Chain the first element of the output of each block to be the input of the next conv_block. 
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0], 4*n_filters)
    cblock4 = conv_block(cblock3[0], 8*n_filters)
    cblock5 = conv_block(cblock4[0], 16*n_filters, max_pooling=False)
    
    
    
    
    