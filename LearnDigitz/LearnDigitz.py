import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import *

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", ".", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", ".", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("log_dir", ".", "Model directory where final model files are saved.")

def main(_):
    # clearing graph
    tf.reset_default_graph()

    # Import MINST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 100
    display_epoch = 1
    unique = datetime.now().strftime('%m-%d_%H_%M')
    logs_path = os.path.join('logs', unique)
    export_path = os.path.join('model', unique)

    # tf Graph Input
    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='InputData')
    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]), name='Weights')
    b = tf.Variable(tf.zeros([10]), name='Bias')

    # Construct model and encapsulating all ops into scopes, making
    # Tensorboard's Graph visualization more convenient
    with tf.name_scope('Model'):
        # Model
        #pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        pred = tf.matmul(x, W) + b # Softmax
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
        # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    
        # Minimize error with *better* cross entropy
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    
        # Minimize error using squared error
        # cost = tf.nn.l2_loss(y - pred)
    
        # Minimize with computed square error
        cost = tf.reduce_mean(tf.pow(y - pred, 2))
    with tf.name_scope('SGD'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

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
    
        # saving model
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(sess, 
                                             [tf.saved_model.tag_constants.SERVING], 
                                             signature_def_map=None,
                                             assets_collection=None)
        builder.save()
        print("Model saved!")
    exit(0)


if __name__ == "__main__":
    tf.app.run()
