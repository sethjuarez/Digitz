import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph as freeze
from tensorflow.examples.tutorials.mnist import input_data
from datetime import *

###################################################################
# Output Paths                                                    #
###################################################################
unique = datetime.now().strftime('%m.%d_%h.%M')
data_path = 'data'
logs_path = os.path.join('logs', 'log_' + unique)
export_path = os.path.join('model', 'model_' + unique)

###################################################################
# Parameters                                                      #
###################################################################
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_epoch = 1

###################################################################
# Models                                                          #
###################################################################
def linear_model(x):
    # set model weights
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')

    # scope for tensorboard
    with tf.name_scope('Model'):
        pred = tf.add(tf.matmul(x, W), b, name="model")  # linear combination
    return pred

def softmax_model(x):
    # set model weights
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')

    # scope for tensorboard
    with tf.name_scope('Model'):
        pred = tf.nn.softmax(tf.matmul(x, W) + b, name="model") # Softmax
    return pred

###################################################################
# Cost / Loss Functions                                           #
###################################################################
def cross_entropy_loss(fn, y):
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(fn), reduction_indices=1))
    return cost

def builtin_cross_entropy_loss(fn, y):
    with tf.name_scope('Loss'):
        # Minimize error with *better* cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fn))
    return cost

def squared_error_loss(fn, y):
    with tf.name_scope('Loss'):
        # Minimize error with *better* cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fn))
    return cost

def builtin_l2_loss(fn, y):
    with tf.name_scope('Loss'):
        # Minimize error using squared error
        cost = tf.nn.l2_loss(y - fn)
    return cost

###################################################################
# Accuracy                                                        #
###################################################################
def get_accuracy(fn, y):
    with tf.name_scope('Accuracy'):
        acc = tf.equal(tf.argmax(fn, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc

###################################################################
# Optimizer                                                       #
###################################################################
def sgd_optimizer(cost, lr):
    with tf.name_scope('SGD'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    return optimizer

###################################################################
# Save Model                                                      #
###################################################################
def save_model(sess):
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
    print("Model saved!")

def main(_):
    # resetting graph
    tf.reset_default_graph()

    # import MINST data
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='input')

    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='label')

    # model
    pred = softmax_model(x)

    # model accuracy
    acc = get_accuracy(pred, y)

    # cost / loss
    cost = cross_entropy_loss(pred, y)

    # trainer
    optimizer = sgd_optimizer(cost, learning_rate)
    

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                         feed_dict={x: batch_xs, y: batch_ys})
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * total_batch + i)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        # Calculate accuracy
        print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

        save_model(sess)
    

    exit(0)


if __name__ == "__main__":
    tf.app.run()
