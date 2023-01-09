# Introduction to TensorFlow for Artificial Intelligence, Machine Leraning, and Deep Learning

Course link:
<https://www.coursera.org/learn/introduction-tensorflow/lecture/>

Sample code:
<https://github.com/lmoroney/dlaicourse>
<https://colab.research.google.com/github/lmoroney/dlaicourse/>

Collection of Interactive Machine Learning Examples
<https://research.google.com/seedbank/>

DeepLearning resources:
<https://www.deeplearning.ai/ai-for-everyone/>
<https://tensorflow.org/>
<https://www.youtube.com/tensorflow>
<http://playground.tensorflow.org/>

## Introduction

``` python
# https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb
import tensorflow as tf
import numpy as np
from tensorflow import keras
# example code to create a NN with 1 neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')  # sgd for stochastic gradient descend
# set input X and expected output Y
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# training
model.fit(xs, ys, epochs=500)
# test
print(model.predict([10.0]))
# [[18.979298]]
```

## Image Recognition Example using Keras and Tensorflow

Repo of the MNIST-like fashion product database
<https://github.com/zalandoresearch/fashion-mnist>
with following features:

* 28x28 grayscale iamges
* label from 10 classes
* 60,000 samples in training set
* 10,000 samples in test set

Reference on Machine Learning Fairness
<https://developers.google.com/machine-learning/fairness-overview/>

``` python
# https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb
import tensorflow as tf

# load the fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])

# normalize pixel values from [0,255] to [0,1]
training_images  = training_images / 255.0
test_images = test_images / 255.0

# set the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(training_images, training_labels, epochs=5)

# testing
model.evaluate(test_images, test_labels)

# a callback to check loss at end of each epoch and cancel trainning is loss < 0.4
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('loss') < 0.4):
      print('\n Loss is low so cancelling training!')
      self.model.stop_training = True

callbacks = myCallback()

# training
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```

### Yet another example to classify handwritten numbers

``` python
import tensorflow as tf

# callback definition
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('loss') < 0.01):
      print('\n Loss is low so cancelling training!')
      self.model.stop_training = True

# load mnist dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# normalize dataset
x_train  = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
callbacks = myCallback()
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# test
model.evaluate(x_test, y_test)
```

### Convolutional Neural Networks

It is used to simplify the image using filters.

Reference:

Keras help docs:

<https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/keras/layers/Conv2D>
<https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/layers/MaxPooling2D>

A video explaination of convolutional neural networks

<https://bit.ly/2UGa7uH>
<https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF>

Image filtering
<https://lodev.org/cgtutor/filtering.html>

Kernal
<https://en.wikipedia.org/wiki/Kernel_(image_processing)>

``` python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)), # 64 3x3 filters
    tf.keras.layers.MaxPooling2D(2, 2), # create max pooling layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu')), # 64 3x3 filters
    tf.keras.layers.MaxPooling2D(2, 2), # create max pooling layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, actication='softmax')
])
model.summary()
```

Notebook - implement a convolution layer and a max pooling layer by hand

<https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb>
<https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb>

``` python
# load a 2D gray image using cv2
import cv2
import numpy as np
from scipy import misc
i = misc.ascent()

# show the image
import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

# create target image by coping the input image
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# create a 3x3 filter
# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

#Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun!
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# If all the digits in the filter don't add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

# create convolution with 1 pixel margin at each side
for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (i[x - 1, y-1] * filter[0][0])
      convolution = convolution + (i[x, y-1] * filter[0][1])
      convolution = convolution + (i[x + 1, y-1] * filter[0][2])
      convolution = convolution + (i[x-1, y] * filter[1][0])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[1][2])
      convolution = convolution + (i[x-1, y+1] * filter[2][0])
      convolution = convolution + (i[x, y+1] * filter[2][1])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      i_transformed[x, y] = convolution

# plot the transformed image
# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()

# a 2x2 pooling for getting largest among 2x2 pixels
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()
```

Notebook - Improving Computer Vision Accuracy using Convolutions

<https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=R6gHiH-I7uFa>
<https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb>

``` python
# do fashion recognition using a Deep Neural Network (DNN) containing three layers
# 1. the input layer (in the shape of the data),
# 2. a hidden layer.
# 3. the output layer (in the shape of the desired output)
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)
# training set: loss: 0.2936 - acc: 0.8918
# testing set: loss: 0.3516 - acc: 0.8715

# same neural network with Convolutional layers added first.
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 26, 26, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 1600)              0
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               204928
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1290
# =================================================================
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)
# training set: loss: 0.1927 - acc: 0.9282
# testing set: loss: 0.2411 - acc: 0.9142

# another setting with 1 conv2D of 32 filters and 1 max pooling
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
# training set: loss: 0.0053 - acc: 0.9981
# testing set: loss: loss: 0.0583 - acc: 0.9873

# Visualizing the Convolutions and Pooling
print(test_labels[:100])
# index 23 and index 28 are all the same value (9). They're all shoes.
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
```

### ImageGenerator

ImageGenerator will generate labels for training set and test set images using directory heirachy, and scale them to same size for training model.

Understand cross entropy:
<https://gombru.github.io/2018/05/23/cross_entropy_loss/>

RMSPropOptimizer:
<https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer>
<http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>

Notebook links:
<https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb>
<https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb>
<https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb>

``` python
# get data from web
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O /tmp/horse-or-human.zip
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip

# unzip
import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
# Directory with our training human pictures
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
# Directory with our training human pictures
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

# display some images
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# ------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
  '/tmp/horse-or-human/',  # This is the source directory for training images
  target_size=(300, 300),  # All images will be resized to 150x150
  batch_size=128,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
  '/tmp/validation-horse-or-human/',  # This is the source directory for training images
  target_size=(300, 300),  # All images will be resized to 150x150
  batch_size=32,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode='binary')

# set up the model, due to the complexity of the horse/human images,
# we use 5 convolutional layers with maxing pooling layers
model = tf.keras.models.Sequential([
  # Note the input shape is the desired size of the image 300x300 with 3 bytes color
  # This is the first convolution
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  # The second convolution
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The third convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The fourth convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The fifth convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # Flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  # 512 neuron hidden layer
  tf.keras.layers.Dense(512, activation='relu'),
  # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_5 (Conv2D)            (None, 298, 298, 16)      448
# _________________________________________________________________
# max_pooling2d_5 (MaxPooling2 (None, 149, 149, 16)      0
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 147, 147, 32)      4640
# _________________________________________________________________
# max_pooling2d_6 (MaxPooling2 (None, 73, 73, 32)        0
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 71, 71, 64)        18496
# _________________________________________________________________
# max_pooling2d_7 (MaxPooling2 (None, 35, 35, 64)        0
# _________________________________________________________________
# conv2d_8 (Conv2D)            (None, 33, 33, 64)        36928
# _________________________________________________________________
# max_pooling2d_8 (MaxPooling2 (None, 16, 16, 64)        0
# _________________________________________________________________
# conv2d_9 (Conv2D)            (None, 14, 14, 64)        36928
# _________________________________________________________________
# max_pooling2d_9 (MaxPooling2 (None, 7, 7, 64)          0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 3136)              0
# _________________________________________________________________
# dense_2 (Dense)              (None, 512)               1606144
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 513
# =================================================================
# Total params: 1,704,097
# Trainable params: 1,704,097
# Non-trainable params: 0

# instead of using model.fit, we use binary cross entropy and model.fit_generator instead
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
  optimizer=RMSprop(lr=0.001),
  metrics=['acc'])

# training
history = model.fit_generator(
  train_generator,
  steps_per_epoch=8,
  epochs=15,
  validation_data=validation_generator,
  validation_steps=8,
  verbose=2)

# run the model
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0] > 0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")

# Visualizing Intermediate Representations
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
```
