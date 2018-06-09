import tensorflow as tf

class ModeParams(object):
    def __init__(self):
        self.epochs = 200
        self.learning_rate = 1e-3
        self.dropout=0.6