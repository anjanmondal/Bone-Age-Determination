# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:22:26 2022

@author: anjan
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Model


def conv_block(tensor_input, filters,max_pooling=True):
    conv= Conv2D(filters=filters,padding='valid',
    kernel_size=(3,3),kernel_initializer='he_normal',activation='relu')(tensor_input)
    
    conv=Conv2D(filters=filters,padding='valid',
    kernel_size=(3,3),kernel_initializer='he_normal',activation='relu')(conv)
    
    
    if max_pooling:
        next_layer = MaxPooling2D((2, 2))(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer,skip_connection

def upsampling_block(prev_input, skip_input, n_filters=64):
    up=Conv2DTranspose(n_filters,kernel_size=(2,2),strides=(2,2))
    
    merge = Concatenate([up, skip_input], axis=3)(prev_input)
    
    conv=Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu',
                kernel_initializer='he_normal')(merge)
    
    conv=Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu',
                kernel_initializer='he_normal')(conv)
    
    
    return conv
    
def unet_model(input_size,n_classes,n_filters=64):
    inputs = Input(input_size)
    cblock1 = conv_block(inputs,n_filters)
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0], 4*n_filters)
    cblock4 = conv_block(cblock3[0], 8*n_filters)
    cblock5 = conv_block(cblock4[0], 16*n_filters, max_pooling=False)
    
    
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters*8)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)


    conv9 = Conv2D(n_filters,
                 kernel_size=(3,3),
                 activation='relu',
                 kernel_initializer='he_normal')(ublock9)

    #conv10 = Conv2D(filters=n_classes, kernel_size=(1,1))(conv9)    
    model = Model(inputs=inputs, outputs=conv9)
    return model
