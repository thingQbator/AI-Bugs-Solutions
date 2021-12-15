Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # -*- coding: utf-8 -*-
"""AI Genarates Art GAN
"""
import requests
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
download_url("https://tqbsandbox.s3.ap-south-1.amazonaws.com/dataset/Abstract_gallery/Abstract_gallery.zip","Abstract_gallery.zip")

import zipfile
with zipfile.ZipFile("Abstract_gallery.zip","r") as zip_ref:
    zip_ref.extractall(".")

# Import numpy for vectorized calculations
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
import numpy as np
# Keras and Tensorflow for Deep Learning 
from keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
# Matplotlib for visualization
from matplotlib import pyplot

# Os for  OS related operations like creating directorires, listing files from a directory
import os
# pillow for Image manipulation
from PIL import Image

# Creating the sigmoid function (for activation)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Creating an Inverse Sigmoid
def inv_sigmoid(x):
    return np.log(y/(1-y))
# Path to the dataset
path='./Abstract_gallery/Abstract_gallery/';
# List images in the source directory
img_list = os.listdir(path)

def access_images(img_list,path,length):
    '''
      Function to read all images from the directory
    '''
    pixels = []
    imgs = []
    for i in range(length):
        # Loop on list of files in the directory
        # Open image using pillow Image
        img = Image.open(path+'/'+img_list[i],'r')
        # Resize image to 100 X 100
        basewidth = 100
        # Cleaning image edges using antialias
        img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
        # Creating array out of image
        pix = np.array(img.getdata())
        # Resize image array to 100 X 100 X 3
        pixels.append(pix.reshape(100,100,3))
        # Append image to list
        imgs.append(img)
    return np.array(pixels),imgs
  
def show_image(pix_list):
	'''
	  Function to display image
	'''
	array = np.array(pix_list.reshape(100,100,3), dtype=np.uint8)
	new_image = Image.fromarray(array)
	new_image.show()

pixels,imgs = access_images(img_list,path,1000)
# Shape of the dataset
print("Shape of the Dataset : ")
print(pixels.shape)

def define_discriminator(in_shape = (100,100,3)):
	'''
	  Function to create descriminator model
	'''
	# Set model to be sequential
	model = Sequential()
	# Add a Convolution 2D layer with 64 filters and a kernel of 3 X 3 and strides 2X2 and padding such that the size remains the same
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	# Adding a LeakyRelu for Activation
	model.add(LeakyReLU(alpha=0.2))
	# Adding a Dropout layer to avoid overfitting
	model.add(Dropout(0.4))
	# Repeat above layers
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	# Flatten to add a fully connected layer
	model.add(Flatten())
	# Add a Fully connected layer
	model.add(Dense(1, activation='sigmoid'))
	# Using Adam optimizer
	opt = Adam(lr=0.0002, beta_1=0.5)
	# Compile the model with binary_crossentropy as loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_generator(latent_dim):
	'''
	  Define the generator model
	'''
    # Set model to be sequential
	model = Sequential()
    # using 25 X 25 X 128 nodes
	n_nodes = 128 * 25 * 25
    # Creating a fully connected network
	model.add(Dense(n_nodes, input_dim=latent_dim))
   	# Adding a leakyRelu for activation
	model.add(LeakyReLU(alpha=0.2))
    # Reshape to 3D
	model.add(Reshape((25, 25, 128)))
    # Use conv2DTranspose remember this is a decoder
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(3, (7,7) , padding='same'))
	return model

def define_gan(g_model, d_model):
    '''
      Create a GAN by adding generator with Descriminator
    '''
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def generate_real_samples(dataset, n_samples):
    '''
      Get real samples from the original dataset
    '''
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y
 
def generate_latent_points(latent_dim, n_samples):
    '''
      To generate a latent space of required size
    '''
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
def generate_fake_samples(g_model, latent_dim, n_samples):
    '''
      Generate fake samples from the generated latent space
    '''
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=10):
    '''
      Function to train the GAN
    '''
    # Calculate batch per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    print(dataset.shape[0])
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        # Iterate on number of epochs
        # In each epoch
        for j in range(bat_per_epo):
            # For each batch
            # Generate Real Samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # Predict fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # Stack Fake on Real samples
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # Calculate Discriminator Model Loss
            d_loss, _ = d_model.train_on_batch(X, y)
            # Calculate Generator Loss
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    '''
      Summary for model performance
    '''
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # Accuracy for real images and fake images
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    # Save Model for every checkpoint
    g_model.save(filename)

# Use the model to generate fake art
# Using a latent dim of 100 X 100
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
print("Shape : ")
print(pixels.shape)
train(g_model, d_model, gan_model, np.array(pixels), latent_dim,n_epochs=1)

from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
model = g_model
latent_points = generate_latent_points(100,1)
X = model.predict(latent_points)
array = np.array(X.reshape(100,100,3), dtype=np.uint8)
new_image = Image.fromarray(array)
new_image.show()
new_image.save("test.jpeg")
