import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob as glob
import albumentations as A
import requests
import zipfile

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, Input, MaxPool2D, Conv2DTranspose
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass
block_plot = False
plt.rcParams['image.cmap'] = 'gray'

def system_config():

@dataclass(frozen=True)
class DatasetConfig:

@dataclass(frozen=True)
class TrainingConfig:

@dataclass(frozen=True)
class InferenceConfig:

def fcn32s_vgg16(num_classes,shape):
    model_input = Input(shape=shape)

    # convolution block 
    x = Conv2D(64,(3,3),padding="same")(model_input)  
    x = Activation("relu")(x)
    x = Conv2D(64,(3,3),padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2,2),strides = (2,2))(x)

    # convolution block 
    x = Conv2D(128,(3,3),padding="same")(model_input)  
    x = Activation("relu")(x)
    x = Conv2D(128,(3,3),padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2,2),strides = (2,2))(x)

    # convolution block
    x = Conv2D(256,(3,3),padding="same")(model_input)  
    x = Activation("relu")(x)
    x = Conv2D(256,(3,3),padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(256,(3,3),padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2,2),strides = (2,2))(x)

    # convolution block 
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # convolution block 
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(num_classes, (1, 1), padding='same')(x)

    outputs = Conv2DTranspose(num_classes, kernel_size=(64, 64), strides=(32, 32),  padding='same')(x)

    model_output = Activation('softmax')(outputs)

    model = Model(inputs=model_input, outputs=model_output)

    return model 
    
    

model = fcn32s_vgg16()
model.summary()

def download_file():

def unzip():

save_name 

class CustomSegDataLoader():
    def __init__():
    def __len__():
    def transforms():
    def resize():
    def reset_array():
    def __getitem__():

def rgb_to_onehot():

def num_to_rgb():

def image_overlay():

def display_image_and_mask():

def create_datasets():

model.compile()

history = model.fit()

def plot_results():


    
