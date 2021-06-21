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
print("x_train.shape:", x_train.shape)

# FLATTENING THE DATA
N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)
x_val = x_val.reshape(-1, D)

# GIVING THE DIMENSIONALITY OF LATENT SPACE
latent_dim = 100

# GENERATOR MODEL
def generator_model(latent_dim):
    
    i = tf.keras.layers.Input(shape=(latent_dim,))
    o = tf.keras.layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha = 0.2))(i)
    o = tf.keras.layers.BatchNormalization(momentum = 0.7)(o)
    o = tf.keras.layers.Dense(512, activation = tf.keras.layers.LeakyReLU(alpha = 0.2))(o)
    o = tf.keras.layers.BatchNormalization(momentum = 0.7)(o)
    o = tf.keras.layers.Dense(1024, activation = tf.keras.layers.LeakyReLU(alpha = 0.2))(o)
    o = tf.keras.layers.BatchNormalization(momentum = 0.7)(o)
    o = tf.keras.layers.Dense(1,activation = 'tanh')(o)
    model = tf.keras.models.Model(i,o)
    return model

# DISCRIMINATOR MODEL
def discriminator_model(img_size):
    
    i = tf.keras.layers.Input(shape=(img_size,))
    o = tf.keras.layers.Dense(512, activation = tf.keras.layers.LeakyReLU(alpha = 0.2))(i)
    o = tf.keras.layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha = 0.2))(o)
    o = tf.keras.layers.Dense(1, activation = 'sigmoid')(o)
    model = tf.keras.models.Model(i,o)
    return model


# COMPILING MODELS FOR TRAINING
discriminator = discriminator_model(D)
discriminator.compile(loss = 'binary_crossentropy',optimizer = tf.keras.optimizers.Adam(0.0002,0.5),metrics = ['accuracy'])

generator = generator_model(latent_dim)
z = tf.keras.layers.Input(shape=(latent_dim,))
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

def sample_images(epoch):
  rows, cols = 5, 5
  noise = np.random.randn(rows * cols, latent_dim)
  imgs = generator.predict(noise)

  imgs = 0.5 * imgs + 0.5

  fig, axs = plt.subplots(rows, cols)
  idx = 0
  for i in range(rows):
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig("gan_images/%d.png" % epoch)
  plt.close()


for epoch in range(EPOCHS):

  idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
  real_imgs = x_train[idx]
  
  noise = np.random.randn(BATCH_SIZE, latent_dim)
  fake_imgs = generator.predict(noise)

  d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
  d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
  d_loss = 0.5 * (d_loss_real + d_loss_fake)
  d_acc  = 0.5 * (d_acc_real + d_acc_fake)
  

  
  noise = np.random.randn(BATCH_SIZE, latent_dim)
  g_loss = combined_model.train_on_batch(noise, ones)
  

  noise = np.random.randn(BATCH_SIZE, latent_dim)
  g_loss = combined_model.train_on_batch(noise, ones)
  

  d_losses.append(d_loss)
  g_losses.append(g_loss)
  
  if epoch % 100 == 0:
    print(f"epoch: {epoch+1}/{EPOCHS}, d_loss: {d_loss:.2f}, \
      d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
  
  if epoch % SAMPLE_PERIOD == 0:
    sample_images(epoch)

