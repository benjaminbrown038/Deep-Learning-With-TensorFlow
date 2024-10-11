import numpy as np
import matplotlib.pyplot as plt 
import os 
import cv2
import glob as glob
import albumentations as A
import requests 
import zipfile

import tensorflow as tf
from tensorflow.keras.models.import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Activation, Dropout, concatenate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass

block_plot = False
plt.rcParams['image.cmap'] = 'gray'


def unet(num_classes,shape):
    model_input = Input(shape)
    
