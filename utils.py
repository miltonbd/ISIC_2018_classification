import sys
import os

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def progress_bar(progress, count ,message):
    sys.stdout.write('\r' + "{} of {}: {}".format(progress, count, message))