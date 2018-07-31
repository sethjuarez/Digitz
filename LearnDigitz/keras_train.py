import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from misc.keras_helpers import export_h5_to_pb
from misc.helpers import print_info, print_args, check_dir
from tensorflow.keras.layers import Reshape, Flatten, Dense, Conv2D, MaxPooling2D

def load_digits(data_dir):
  mnist = tf.keras.datasets.mnist
  path = os.path.join(data_dir, 'mnist.npz')
  (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) / 255.0
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]) / 255.0
  y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
  return (x_train, y_train),(x_test, y_test)
###################################################################
# Simple (W.T * X + b)                                            #
###################################################################
def linear():
  model = Sequential([Dense(10)])
  return model

###################################################################
# Neural Network                                                  #
###################################################################
def mlp(hidden=[512, 512]):
  model = Sequential()
  for i in range(len(hidden)):
    model.add(Dense(512, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  return model

###################################################################
# Convolutional Neural Network                                    #
###################################################################
def cnn():
  model = tf.keras.Sequential([
    Reshape((-1, 28, 28, 1)),
    Conv2D(32, [5, 5], padding='same', activation='relu'),
    MaxPooling2D(strides=2),
    Conv2D(64, [5, 5], padding='same', activation='relu'),
    MaxPooling2D(strides=2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')
  ])
  return model

def run(data_dir, model_dir, epochs):
  # get data
  (x_train, y_train), (x_test, y_test) = load_digits(data_dir)

  # create model structure
  model = linear()

  # compile model
  model.compile(loss='mean_squared_error', 
    optimizer='adam', 
    metrics=['accuracy'])

  # run model
  model.fit(x_train, y_train, epochs=epochs)

  model.summary()
  model.evaluate(x_test, y_test)

  m = os.path.join(model_dir, 'model.h5')
  print('Saving model to {}'.format(m))
  model.save(m)
  print('Saving protocol buffer')
  export_h5_to_pb(model, model_dir)

if __name__ == "__main__":
  data_dir = check_dir(os.path.abspath('data'))
  output_dir = os.path.abspath('output')
  unique = datetime.now().strftime('%m.%d_%H.%M')
  model_dir = check_dir(os.path.join(output_dir, 'models', 'model_{}'.format(unique)))
  epochs = 1
  run(data_dir, model_dir, epochs)