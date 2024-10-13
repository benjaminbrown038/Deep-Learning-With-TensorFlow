import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import zipfile
import requests
import albumentations as A 
import cv2
import os
from tensorflow.keras.utils import Sequence
from dataclasses import dataclass

def download_file(url,save_name):
    file = requests.get(url)
    open(save_name,'wb').write(file.content)


def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all")
    except:
        print("Invalid file")


save_name = "dataset_SUIM.zip"

if not os.path.exists(save_name):
    download_file('https://www.dropbox.com/s/1g2y2nu9v7gizu9/dataset_SUIM.zip?dl=1',save_name)
    unzip(zip_file = save_name)

