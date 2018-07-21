import sys
import os
import numpy as np


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_if_exists(dir):
    return os.path.exists(dir)

def progress_bar(progress, count ,message):
    sys.stdout.write('\r' + "{} of {}: {}".format(progress, count, message))