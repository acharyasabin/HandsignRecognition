import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from predict1 import funct
from model.utils import set_logger

name = "predict.log"
for root, dirs, files in os.walk("predict"):
    if name in files:
        os.remove("predict/predict.log")

for x in range(0,6):
    funct(x)


lookup = '1.000'

with open("predict/predict.log") as myFile:
    for num, line in enumerate(myFile, 1):
        if lookup in line:
            print ("The label for the image is ",(int((num/2)-1)))


