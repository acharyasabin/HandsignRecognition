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



def funct(x):
    # Set the random seed for the whole graph

    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)



    # Set the logger
    set_logger(os.path.join(args.data_dir, 'predict.log'))

    # Create the input data pipeline

    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir)

    # Get the filenames from the test set

    test_filenames = [os.path.join(test_data_dir, 'predict.jpg') ]

    test_labels = [x]
    # print(test_labels)

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model

    model_spec = model_fn('eval', test_inputs, params, reuse=tf.AUTO_REUSE)


    evaluate(model_spec, args.model_dir, params, args.restore_from)


