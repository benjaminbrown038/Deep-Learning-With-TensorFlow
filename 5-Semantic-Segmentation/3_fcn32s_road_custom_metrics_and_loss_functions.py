import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import glob as glob 
import albumentations as A
import requests
import zipfile

from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, Input, MaxPool2D, Conv2DTranspose
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass

block_plot = False
plt.rcParams[] = 'gray'

def system_config():

@dataclass(frozen=True)
class DatasetConfig:

@dataclass(frozen=True)
class TrainingConfig:

@dataclass(frozen=True)
class InferenceConfig:

def fcn32s_vgg16(num_classes,shape):
    model_input = Input(shape)
      # Conv block 1.
    x = Conv2D(64, (3, 3), padding='same')(model_input)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Conv block 2.
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Conv block 3.
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Conv block 4.
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Conv block 5.
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    
    # 1x1 convolution to limit the depth of the feature maps to the number of classes.
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    
    # Upsampling using Transposed Convolution.
    outputs = Conv2DTranspose(num_classes, kernel_size=(64, 64), strides=(32, 32),  padding='same')(x)

    model_output = Activation('softmax')(outputs)

    model = Model(inputs=model_input, outputs=model_output)

    return model
  
