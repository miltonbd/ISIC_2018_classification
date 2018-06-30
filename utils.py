import sys
import os
import numpy as np

def get_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return  params

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def progress_bar(progress, count ,message):
    sys.stdout.write('\r' + "{} of {}: {}".format(progress, count, message))