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
from matplotlib.ticker import MutilipleLocator, FormatStrFormatter
from dataclasses import dataclass
block_plot = False

def fcn32s_vgg16(num_classes,shape):
  
    model_input = Input(shape=shape)

    # Conv Block 1 
    x = Conv2D()
    x = Activation()
    x = Conv2D()
    x = Activation()
    x = MaxPool2D()

    # Conv Block 2 
    x = Conv2D()
    x = Activation()
    x = Conv2D()
    x = Activation()
    x = MaxPool2D()

    # Conv Block 3 
    x = Conv2D()
    x = Activation()
    x = Conv2D()
    x = Activation()
    x = Conv2D()
    x = Activation()
    x = MaxPool2D()

    # Conv Block 4 
    x = Conv2D()
    x = Activation()


    x = Conv2D()

    ouputs = Conv2DTranspose()
    model_output = Activation()
    model = Model(inputs = model_input, outputs = model_outputs)
    return model 
