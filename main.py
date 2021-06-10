# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf

# GETTING DATA
data = tf.keras.datasets.mnist
(x_train,y_train),(x_val,y_val) = data.load_data()

x_train,x_val = x_train/255.0*2-1,x_val/255.0*2-1

N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)
x_val = x_val.reshape(-1, D)

