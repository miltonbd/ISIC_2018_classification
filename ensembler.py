import os
saved_model_dir="./saved_models"

def make_ensemble_comitte():
    predictions=[]
    for model_name in os.listdir(saved_model_dir):
        # load a model
        # run all the test images
    # calculate average of all outputs



