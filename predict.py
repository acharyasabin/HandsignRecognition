"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='predict',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

def build_model(image):

    out = image
    #assert image.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    out = tf.reshape(out, shape=[-1, 64, 64, 3])

    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    #assert image.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    #assert out.get_shape().as_list() == [None, 4, 4, num_channels * 8]

    out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 8)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out,params.num_labels)

    return logits
def _parse_function():
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file("predict/predict.jpg")

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [64, 64])

    return resized_image



if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # image = tf.image.decode_jpeg(tf.read_file("predict/predict.jpg"), channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize_images(image, [64, 64])
    # print(image)
    #images = {"images": image,"labels": "asd"}\
    #init_op = tf.global_variables_initializer()


    with tf.Session() as sess:

        # Read meta graph and checkpoint to restore tf session
        saver = tf.train.import_meta_graph("experiments/test/best_weights/after-epoch-10.meta")
        saver.restore(sess, "experiments/test/best_weights/after-epoch-10")
        logits = (build_model(_parse_function()))
        prediction = tf.argmax(logits, 1)
        sess.run(tf.global_variables_initializer())
        # 'x' is what you defined it to be. In my case it is a batch of RGB images, that's why I add the extra dimension
        # (prediction.eval(session=sess))
        print(logits.eval(session=sess))
        # sess.run(tf.global_variables_initializer())
        # # sess.run(tf.global_variables_initializer())
        #
        # # tvars = tf.trainable_variables()
        # # tvars_vals = sess.run(tvars)
        # # for var, val in zip(tvars, tvars_vals):
        # #
        # #     print(var.name, val)
        # sess.run(["model/fc_2/dense/bias:0", "model/block_1/conv2d/kernel:0","model/block_1/conv2d/bias: 0",
        # "model/block_1/batch_normalization/gamma:0",
        # "model/block_1/batch_normalization/beta:0",
        # "model/block_2/conv2d/kernel:0",
        # "model/block_2/conv2d/bias:0",
        # "model/block_2/batch_normalization/gamma:0",
        # "model/block_2/batch_normalization/beta:0",
        # "model/block_3/conv2d/kernel:0",
        # "model/block_3/conv2d/bias: 0",
        # "model/block_3/batch_normalization/gamma:0",
        # "model/block_3/batch_normalization/beta:0",
        # "model/block_4/conv2d/kernel:0",
        # "model/block_4/conv2d/bias: 0",
        # "model/block_4/batch_normalization/gamma:0",
        # "model/block_4/batch_normalization/beta:0",
        # "model/fc_1/dense/kernel:0",
        # "model/fc_1/dense/bias:0",
        # "model/fc_1/batch_normalization/gamma:0",
        # "model/fc_1/batch_normalization/beta:0",
        # "model/fc_2/dense/kernel:0",
        # "model/fc_2/dense/bias:0"])
        #
        # # model / block_1 / conv2d / bias: 0
        # # model / block_1 / batch_normalization / gamma: 0
        # # model / block_1 / batch_normalization / beta: 0
        # # model / block_2 / conv2d / kernel: 0
        # # model / block_2 / conv2d / bias: 0
        # # model / block_2 / batch_normalization / gamma: 0
        # # model / block_2 / batch_normalization / beta: 0
        # # model / block_3 / conv2d / kernel: 0
        # # model / block_3 / conv2d / bias: 0
        # # model / block_3 / batch_normalization / gamma: 0
        # # model / block_3 / batch_normalization / beta: 0
        # # model / block_4 / conv2d / kernel: 0
        # # model / block_4 / conv2d / bias: 0
        # # model / block_4 / batch_normalization / gamma: 0
        # # model / block_4 / batch_normalization / beta: 0
        # # model / fc_1 / dense / kernel: 0
        # # model / fc_1 / dense / bias: 0
        # # model / fc_1 / batch_normalization / gamma: 0
        # # model / fc_1 / batch_normalization / beta: 0
        # # model / fc_2 / dense / kernel: 0
        # # model / fc_2 / dense / bias: 0
        #
        #
        # Substitute 'logits' with your model
        # prediction = tf.argmax(build_model(_parse_function()), 1)
        # print(prediction)
        # 'x' is what you defined it to be. In my case it is a batch of RGB images, that's why I add the extra dimension
        # prediction.eval()
        #
        #





        # predictions = tf.argmax(logits, 1)
        # # tf.get_collection()




        # print(sess.run(((build_model(_parse_function())))))
        # # probabilities = tf.nn.softmax(logits, name='Predictions')
        # # print(probabilities)
        #
        #
        #
        # #eval_model_spec = model_fn('eval', image, params, reuse=True)
        # #predictions = sess.run(model_fn('eval', images, params, reuse=True))






