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

# FLATTENING THE DATA
N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)
x_val = x_val.reshape(-1, D)

# GIVING THE DIMENSIONALITY OF LATENT SPACE
latent_dim = 100

# GENERATOR MODEL
def generator_model(latent_dim):
    model = tf.keras.models.Model()
    model.add(tf.keras.layers.Input(shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha = 0.2)))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Dense(512, activation = tf.keras.layers.LeakyReLU(alpha = 0.2)))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Dense(1024, activation = tf.keras.layers.LeakyReLU(alpha = 0.2)))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Dense(activation = 'tanh'))
    return model

# DISCRIMINATOR MODEL
def discriminator_model(img_size):
    model = tf.keras.models.Model()
    model.add(tf.keras.layers.Input(shape=(img_size,)))
    model.add(tf.keras.layers.Dense(512, activation = tf.keras.layers.LeakyReLU(alpha = 0.2)))
    model.add(tf.keras.layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha = 0.2)))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model


# COMPILING MODELS FOR TRAINING
discriminator = discriminator_model(D)
discriminator.compile(loss = 'binary_crossentropy',optimizer = tf.keras.optimizers.Adam(0.0002,0.5),metrics = ['accuracy'])

generator = generator_model(latent_dim)
z = tf.keras.layers.Input(shape=(latent_dim))
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)

combined_model = tf.keras.models.Model(z,fake_pred)
combined_model.compile(loss = 'binary_crossentropy',optimizer = tf.keras.optimizers.Adam(0.0002,0.5))

# TRAINING THE GAN
BATCH_SIZE = 32
EPOCHS = 30000
SAMPLE_PERIOD = 200

ones = np.ones(BATCH_SIZE)
zeros = np.zeros(BATCH_SIZE)

d_losses = []
g_losses = []

if not os.path.exists('GAN_Images'):
  os.makedirs('GAN_Images')