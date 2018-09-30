import os
name = "predict.log"
for root, dirs, files in os.walk("predict"):
    if name in files:
        os.remove("predict/predict.log")