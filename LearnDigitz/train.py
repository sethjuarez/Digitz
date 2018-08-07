import os
import sys
import argparse
import numpy as np
from tqdm import tqdm, trange
import tensorflow as tf
from datetime import datetime
from misc.digits import Digits
from tensorflow.examples.tutorials.mnist import input_data
from misc.helpers import print_info, print_args, check_dir, info, save_model

###################################################################
# Models                                                          #
###################################################################
@print_info
def linear_model(x):
    with tf.name_scope("Model"):
        # set model weights
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        b = tf.Variable(tf.zeros([10]), name='b')

        # linear combination
        return tf.add(tf.matmul(x, W), b, name="prediction")

@print_info
def mlp_model(x, hidden=[512, 512]):
    # for size of input layer
    hidden.insert(0, 28 * 28)

    last_output = x
    for i in range(len(hidden) - 1):
        # layer n
        with tf.name_scope("Layer" + str(i)):
            W = tf.Variable(tf.keras.initializers.he_uniform([hidden[i], hidden[i+1]]), name="h" + str(i))
            b = tf.Variable(tf.keras.initializers.he_uniform([hidden[i+1]]), name="b" + str(i))
            layer = tf.add(tf.matmul(last_output, W), b, name="layer" + str(i))
            # for next iteration
            last_output = tf.nn.relu(layer, name="activation" + str(i))

    # output layer
    with tf.name_scope("Model"):
        Wo = tf.Variable(tf.keras.initializers.he_uniform([hidden[-1], 10]), name="Wo")
        Bo = tf.Variable(tf.keras.initializers.he_uniform([10]), name="Bo")
        pred = tf.nn.softmax(tf.matmul(last_output, Wo) + Bo, name="prediction")

    return pred

@print_info
def cnn_model(x):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = tf.nn.relu(conv1 + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(conv2 + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('Model'):
        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        
        y_conv = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="prediction")

    return y_conv

###################################################################
# Models (Better)                                                 #
###################################################################
@print_info
def linear_better(x, init=tf.zeros):
    with tf.name_scope("Model"):
        pred = tf.layers.dense(inputs=x, units=10)
        return tf.identity(pred, name="prediction")

@print_info
def mlp_better(x):
    # hidden layers
    h1 = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=512, activation=tf.nn.relu)

    # output layer
    with tf.name_scope("Model"):
        pred = tf.layers.dense(inputs=h2, units=10, activation=tf.nn.softmax)
        return tf.identity(pred, name="prediction")

@print_info
def cnn_better(x):
    conv1 = tf.layers.conv2d(inputs=tf.reshape(x, [-1, 28, 28, 1]), 
                             filters=32, 
                             kernel_size=[5, 5], 
                             padding="same", 
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    with tf.name_scope('Model'):
        pred = tf.layers.dense(inputs=dense, units=10, activation=tf.nn.softmax)
        return tf.identity(pred, name="prediction")

###################################################################
# Training                                                        #
###################################################################
@print_info
def train_model(x, y, cost, optimizer, accuracy, learning_rate, batch_size, epochs, data_dir, model_dir, log_dir):
    info('Initializing Devices')
    print(' ')
    
    # load MNIST data (if not available)
    digits = Digits(data_dir, batch_size)
    test_x, test_y = digits.test
    
    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    # Initializing the variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        acc = 0.
        info('Training')
        # epochs to run
        with trange(epochs, desc="{:<10}".format("Training"), bar_format='{l_bar}{bar}|{postfix}', postfix=" acc: 0.0000") as t:
            for epoch in t:
                avg_cost = 0.
                t.postfix = ' acc: {:.4f}'.format(acc)
                t.update()
                # loop over all batches
                with tqdm(enumerate(digits), total=digits.total, desc="{:<10}".format("Epoch {}".format(epoch + 1)), bar_format='{l_bar}{bar}|{postfix}') as progress:
                    for i, (train_x, train_y) in progress:
                        # Run optimization, cost, and summary
                        _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                                feed_dict={x: train_x, y: train_y})

                        # Write logs at every iteration
                        summary_writer.add_summary(summary, epoch * digits.total + i)
                        # Compute average loss
                        avg_cost += c / digits.total
                        progress.postfix = 'loss: {:.4f}'.format(avg_cost)
                        progress.update()

                acc = accuracy.eval({x: test_x, y: test_y})

        print('\n\nFinal Accuracy: {:.4f}'.format(acc))
        
        # save model
        info("Saving Model")
        save_model(sess, model_dir, 'Model/prediction')

def main(settings):
    # resetting graph
    tf.reset_default_graph()

    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # model
    hx = linear_model(x)

    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hx, 1), tf.argmax(y, 1)), tf.float32))

    # cost / loss
    cost = tf.reduce_mean(tf.pow(hx - y, 2))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(settings.lr).minimize(cost)

    # training session
    train_model(x, y, cost, optimizer, accuracy, 
        settings.lr, settings.batch, settings.epochs, 
        settings.data, settings.model, settings.log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-o', '--output', help='output directory', default='output')
    args = parser.parse_args()

    args.data = check_dir(os.path.abspath(args.data))
    args.output = os.path.abspath(args.output)
    unique = datetime.now().strftime('%m.%d_%H.%M')
    args.log = check_dir(os.path.join(args.output, 'logs', 'log_{}'.format(unique)))
    args.model = check_dir(os.path.join(args.output, 'models', 'model_{}'.format(unique)))
    
    main(args)
