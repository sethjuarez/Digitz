import numpy as np
import inspect
import sys
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph as freeze
from tensorflow.examples.tutorials.mnist import input_data
from datetime import *

###################################################################
# Parameters                                                      #
###################################################################
learning_rate = 0.01
training_epochs = 10
batch_size = 100

###################################################################
# Output Paths / Utils                                            #
###################################################################
unique = datetime.now().strftime('%m.%d_%H.%M')
data_path = 'data'
logs_path = os.path.join('output', 'logs', 'log_' + unique)
export_path = os.path.join('output', 'model', 'model_' + unique)

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def info_caller():
    # getting calling function
    caller = inspect.stack()[1]
    print(" - Using: %s" % caller.function)
    args, _, _, values = inspect.getargvalues(caller[0])
    for i in args:
        o = str(values[i]).replace("\n", "            \n")
        print ("      %s = %s" % (i, o))
    
###################################################################
# Save Model                                                      #
###################################################################
def save_model(sess, export_path):
    # saving model
    checkpoint = os.path.join(export_path, "model.ckpt")
    saver = tf.train.Saver()
    # checkpoint - variables
    saver.save(sess, checkpoint)
    # graph
    tf.train.write_graph(sess.graph_def, export_path, "model.pb", as_text=False)
    # freeze
    g = os.path.join(export_path, "model.pb")
    frozen = os.path.join(export_path, "digits.pb")
        
    freeze.freeze_graph(
        input_graph = g, 
        input_saver = "", 
        input_binary = True, 
        input_checkpoint = checkpoint, 
        output_node_names = "Model/model",
        restore_op_name = "",
        filename_tensor_name = "",
        output_graph = frozen,
        clear_devices = True,
        initializer_nodes = ""
    )
    print("Model saved to " + frozen)

###################################################################
# Models                                                          #
###################################################################
def linear_model(x, init=tf.zeros):
    info_caller()
    # set model weights
    W = tf.Variable(init([784, 10]), name='weights')
    b = tf.Variable(init([10]), name='bias')

    # scope for tensorboard
    with tf.name_scope('Model'):
        pred = tf.add(tf.matmul(x, W), b, name="model")  # linear combination
    return pred

def softmax_model(x, init=tf.zeros):
    info_caller()
    # set model weights
    W = tf.Variable(init([784, 10]), name='weights')
    b = tf.Variable(init([10]), name='bias')

    # scope for tensorboard
    with tf.name_scope('Model'):
        pred = tf.nn.softmax(tf.matmul(x, W) + b, name="model") # Softmax
    return pred

def multilayer_perceptron_model(x, init=tf.random_normal, hidden = [256, 256]):
    info_caller()

    # for size of input layer
    hidden.insert(0, 28 * 28)

    last_output = x
    for i in range(len(hidden) - 1):
        # layer n
        with tf.name_scope("Layer" + str(i)):
            W = tf.Variable(init([hidden[i], hidden[i+1]]), name="h" + str(i))
            b = tf.Variable(init([hidden[i+1]]), name="b" + str(i))
            layer = tf.add(tf.matmul(last_output, W), b, name="layer" + str(i))
            # for next iteration
            last_output = layer

    # output layer
    with tf.name_scope("Model"):
        Wo = tf.Variable(init([hidden[-1], 10]), name="Wo")
        Bo = tf.Variable(init([10]), name="Bo")
        pred = tf.add(tf.matmul(last_output, Wo), Bo, name="model")

    return pred

def multilayer_perceptron_relu_softmax_model(x, init=tf.random_normal, hidden=[256, 256]):
    info_caller()

    # for size of input layer
    hidden.insert(0, 28 * 28)

    last_output = x
    for i in range(len(hidden) - 1):
        # layer n
        with tf.name_scope("Layer" + str(i)):
            W = tf.Variable(init([hidden[i], hidden[i+1]]), name="h" + str(i))
            b = tf.Variable(init([hidden[i+1]]), name="b" + str(i))
            layer = tf.add(tf.matmul(last_output, W), b, name="layer" + str(i))
            # for next iteration
            last_output = tf.nn.relu(layer, name="activation" + str(i))

    # output layer
    with tf.name_scope("Model"):
        Wo = tf.Variable(init([hidden[-1], 10]), name="Wo")
        Bo = tf.Variable(init([10]), name="Bo")
        pred = tf.nn.softmax(tf.matmul(last_output, Wo) + Bo, name="model")

    return pred

def convolutional_neural_network_model(x):
    info_caller()
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
        
        y_conv = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="model")

    return y_conv

###################################################################
# Cost / Loss Functions                                           #
###################################################################
def cross_entropy_loss(fn, y):
    info_caller()
    with tf.name_scope('loss'):
        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(fn), reduction_indices=1))
    return cost

def builtin_cross_entropy_loss(fn, y):
    info_caller()
    with tf.name_scope('loss'):
        # Minimize error with *better* cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fn))
    return cost

def sparse_softmax_cross_entropy_loss(fn, y):
    info_caller()
    with tf.name_scope('loss'):
        cost = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=fn)
        cost = tf.reduce_mean(cost)
    return cost

def squared_error_loss(fn, y):
    info_caller()
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.pow(y - fn, 2))
    return cost

def builtin_l2_loss(fn, y):
    info_caller()
    with tf.name_scope('loss'):
        # Minimize error using squared error
        cost = tf.nn.l2_loss(y - fn)
    return cost

###################################################################
# Accuracy                                                        #
###################################################################
def get_accuracy(fn, y):
    with tf.name_scope('score'):
        acc = tf.equal(tf.argmax(fn, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc

###################################################################
# Optimizer                                                       #
###################################################################
def sgd_optimizer(cost, lr):
    info_caller()
    with tf.name_scope('optimizer'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    return optimizer

def adam_optimizer(cost, lr):
    info_caller()
    with tf.name_scope('optimizer'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    return optimizer

###################################################################
# Train Model                                                     #
###################################################################
def train_model(optimizer, cost, accuracy, x, y, batch_size = 100, training_epochs = 10):
    info("Training Phase")
    info_caller()
    # import MINST data
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Initializing the variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization, cost, and summary
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                          feed_dict={x: batch_xs, y: batch_ys})
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * total_batch + i)
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Calculate accuracy
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        info("Saving Model")
        # save model
        save_model(sess, export_path)

###################################################################
# Main                                                            #
###################################################################
def main(_):
    # resetting graph
    tf.reset_default_graph()

    info("Building Computation Graph")

    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='input')

    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='label')

    # model 
    #predictor = multilayer_perceptron_relu_softmax_model(x, 
    #                                init=tf.keras.initializers.he_uniform())
    predictor = convolutional_neural_network_model(x)

    # model accuracy
    accuracy = get_accuracy(predictor, y)

    # cost / loss
    cost = builtin_cross_entropy_loss(predictor, y)

    # optimizer
    optimizer = adam_optimizer(cost, learning_rate)

    # training
    train_model(optimizer, cost, accuracy, x, y, batch_size, training_epochs)

    exit(0)

if __name__ == "__main__":
    tf.app.run()
