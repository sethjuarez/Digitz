import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from misc.helpers import print_info, print_args, check_dir, info
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.keras.layers import Reshape, Flatten, Dense, Conv2D, MaxPooling2D

def save(model, model_dir):
  m = os.path.join(model_dir, 'model.h5')
  print('\nSaving h5 model to {}'.format(m))
  model.save(m)
  print('Saving pb model to {}'.format(os.path.join(model_dir, 'digits.pb')))
  
  input_node = model.input.name.split(':')[0]
  output_node = model.output.name.split(':')[0]
  print("\nInput Tensor:", input_node)
  print("Output Tensor:", output_node)
  
  K.set_learning_phase(0)
  with K.get_session() as sess:
    graph = graph_util.convert_variables_to_constants(sess, 
      sess.graph.as_graph_def(), [output_node])
    graph_io.write_graph(graph, model_dir, 'digits.pb', as_text=False)

def load_digits(data_dir):
  mnist = tf.keras.datasets.mnist
  path = os.path.join(data_dir, 'mnist.npz')
  (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) / 255.0
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]) / 255.0
  y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)

###################################################################
# shapes                                                          #
###################################################################
def linear():
  return Sequential([Dense(10)])

def mlp():
  return Sequential([
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
  ])

def cnn():
  return Sequential([
    Reshape((28, 28, 1)),
    Conv2D(32, [5, 5], padding='same', activation='relu'),
    MaxPooling2D(strides=2),
    Conv2D(64, [5, 5], padding='same', activation='relu'),
    MaxPooling2D(strides=2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')
  ])


@print_info
def run(data_dir, model_dir, epochs):
  # get data
  (x_train, y_train), (x_test, y_test) = load_digits(data_dir)
  
  # create model structure
  model = cnn()
  
  # compile model
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  
  # run model
  model.fit(x_train, y_train, epochs=epochs)
  model.summary()
  evaluation = model.evaluate(x_test, y_test)
  
  # save model
  info('Output...')
  save(model, model_dir)

  # metrics
  info('Metrics...')
  print('Loss:     {}'.format(evaluation[0]))
  print('Accuracy: {}'.format(evaluation[1]))
 

if __name__ == "__main__":
  data_dir = check_dir(os.path.abspath('data'))
  output_dir = os.path.abspath('output')
  unique = datetime.now().strftime('%m.%d_%H.%M')
  model_dir = check_dir(os.path.join(output_dir, 'models', 'model_{}'.format(unique)))
  epochs = 10
  run(data_dir, model_dir, epochs)