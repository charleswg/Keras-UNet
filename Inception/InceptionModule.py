'''
Created on Aug 4, 2019

@author: daniel
Modified for GConv by Kelvin Wong (2021)
'''
from keras.layers import Activation, MaxPooling2D, concatenate

from keras_gcnn.layers import GConv2D, GBatchNorm
from keras_gcnn.layers.pooling import GroupPool
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util, gconv2d

def InceptionModule(inputs, numFilters = 32, h_input=None, h_output=None):
    weight_decay=0
       
    tower_0 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_0 = GBatchNorm(h_output)(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_1 = GBatchNorm(h_output)(tower_1)
    tower_1 = Activation("relu")(tower_1)
    tower_1 = GConv2D(numFilters, (3, 3), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_1)
    tower_1 = GBatchNorm(h_output)(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_2 = GBatchNorm(h_output)(tower_2)
    tower_2 = Activation("relu")(tower_2)
    tower_2 = GConv2D(numFilters, (3, 3), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_2)
    tower_2 = GConv2D(numFilters, (3, 3), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_2)
    tower_2 = GBatchNorm(h_output)(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_3 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_3)
    tower_3 = GBatchNorm(h_output)(tower_3)
    tower_3 = Activation("relu")(tower_3)
    
    inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    return inception_module


def DilatedInceptionModule(inputs, numFilters = 32, h_input=None, h_output=None): 
    tower_0 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_0 = GBatchNorm(h_output)(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_1 = GBatchNorm(h_output)(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=h_output,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_2 = GBatchNorm(h_output)(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    dilated_inception_module = concatenate([tower_0, tower_1, tower_2], axis = 3)
    return dilated_inception_module
