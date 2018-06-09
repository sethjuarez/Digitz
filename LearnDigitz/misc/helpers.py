import os
import time
import tensorflow as tf
from functools import wraps
from inspect import getargspec
# pylint: disable-msg=E0611
from tensorflow.python.tools import freeze_graph as freeze
# pylint: enable-msg=E0611

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def print_info(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        info('-> {}'.format(f.__name__))
        print('Parameters:')
        ps = list(zip(getargspec(f).args, args))
        width = max(len(x[0]) for x in ps) + 1
        for t in ps:
            items = str(t[1]).split('\n')
            print('   {0:<{w}} ->  {1}'.format(t[0], items[0], w=width))
            for i in range(len(items) - 1):
                print('   {0:<{w}}       {1}'.format(' ', items[i+1], w=width))
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print('\n -- Elapsed {0:.4f}s\n'.format(te-ts))
        return result
    return wrapper

def print_args(args):
    info('Arguments')
    ps = args.__dict__.items()
    width = max(len(k) for k, _ in ps) + 1
    for k, v in ps:
        print('   {0:<{w}} ->  {1}'.format(k, str(v), w=width))

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

@print_info
def test_func(a, b, c, d, e, f):
    print('--------From Test----------')
    print(a, b, c, d, e, f)

def save_model(sess, export_path, output_node):
    # saving model
    checkpoint = os.path.join(export_path, "model.ckpt")
    saver = tf.train.Saver()
    # checkpoint - variables
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        
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
        output_node_names = output_node,
        restore_op_name = "",
        filename_tensor_name = "",
        output_graph = frozen,
        clear_devices = True,
        initializer_nodes = "")

    print("Model saved to " + frozen)

if __name__ == "__main__":
    test_func(1, 2, 3, 4, 5, 6)