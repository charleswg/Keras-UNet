'''
Created on Aug 4, 2019

@author: daniel
Modified for GConv by Kelvin Wong (2021)
'''
from keras.layers import Activation, MaxPooling2D, concatenate

from keras_gcnn.layers import GConv2D, GBatchNorm
from keras_gcnn.layers.pooling import GroupPool
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util, gconv2d

def InceptionModule(self, inputs, numFilters = 32, h_input=None, h_output=None):
    conv_group = self.conv_group
    weight_decay=self.weight_decay
    if h_input is None:
        h_input= conv_group
    if h_output is None:
        h_output= conv_group
        
    tower_0 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_0 = GBatchNorm(conv_group)(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_1 = GBatchNorm(conv_group)(tower_1)
    tower_1 = Activation("relu")(tower_1)
    tower_1 = GConv2D(numFilters, (3, 3), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_1)
    tower_1 = GBatchNorm(conv_group)(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(inputs)
    tower_2 = GBatchNorm(conv_group)(tower_2)
    tower_2 = Activation("relu")(tower_2)
    tower_2 = GConv2D(numFilters, (3, 3), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_2)
    tower_2 = GConv2D(numFilters, (3, 3), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_2)
    tower_2 = GBatchNorm(conv_group)(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_3 = GConv2D(numFilters, (1, 1), padding='same',kernel_initializer = 'he_normal',h_input=h_input,h_output=conv_group,bias_initializer=Constant(0.0001), kernel_regularizer=l2(weight_decay))(tower_3)
    tower_3 = GBatchNorm(conv_group)(tower_3)
    tower_3 = Activation("relu")(tower_3)
    
    inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    return inception_module


def DilatedInceptionModule(inputs, numFilters = 32): 
    tower_0 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (1,1), kernel_initializer = 'he_normal')(inputs)
    tower_0 = BatchNormalization()(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (2,2), kernel_initializer = 'he_normal')(inputs)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (3,3), kernel_initializer = 'he_normal')(inputs)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    dilated_inception_module = concatenate([tower_0, tower_1, tower_2], axis = 3)
    return dilated_inception_module
