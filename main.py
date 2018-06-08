import os
from torch import nn,optim
import torchvision
from classifier import Classifier
from pretrainedmodels.models import inceptionv4
from augment_data import augment_images

"""
We trained an Inception V3 network for the three class task. 
The overall classification accuracy is 0.65. Class-wise accuracy is 0.72 (Positive),  0.60 (Neutral) and 0.60 (Negative).
"""

class ModelDetailsInceptionV4(object):

    def __init__(self):
        model = inceptionv4()
        model.avg_pool = nn.AvgPool2d(5, count_include_pad=False)
        model.last_linear = nn.Linear(1536, 3)

        # todo freeze few layers in first
        # todo augement data set and use random crop, pair augment
        # todo mix the emotional images like image pair
        # todo cut the face and add another emotion
        # todo add random
        # todo new loss function
        # todo new optimization
        # todo new training proceudres

        self.model= model
        self.learning_rate = 0.001
        self.eps=200
        self.optimizer = "adam"
        self.model_name_str = "inceptionv4"
        self.batch_size=30
        self.epochs = 200
        self.logs_dir  = "logs/inceptionv4/no_aug"
        self.augmentor = augment_images

model_details=ModelDetailsInceptionV4()
model_details.model_name= "inceptionv4"

clasifier=Classifier(model_details)
clasifier.load_data()
clasifier.load_model()
for epoch in range(clasifier.start_epoch, clasifier.start_epoch + model_details.epochs):
    try:
      clasifier.train(epoch)
      clasifier.test(epoch)
    except KeyboardInterrupt:
      clasifier.test(epoch)
      break;
    clasifier.load_data()
