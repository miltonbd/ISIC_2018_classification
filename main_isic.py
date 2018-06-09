import os
from torch import nn,optim
import torchvision
from classifier_isic import Classifier
from pretrainedmodels.models.inceptionv4 import  inceptionv4
from augment_data_isic import augment_images
import time
"""
We trained an Inception V3 network for the three class task. 
The overall classification accuracy is 0.65. Class-wise accuracy is 0.72 (Positive),  0.60 (Neutral) and 0.60 (Negative).
sudo nvidia-smi -i 0 -pl 180
"""
# import  os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ModelDetailsInceptionV4(object):

    def __init__(self):
        model = inceptionv4()
        model.avg_pool = nn.AvgPool2d(5, count_include_pad=False)
        model.last_linear = nn.Linear(1536, 7)

        # todo freeze few layers in first
        # todo augement data set and use random crop, pair augment
        # todo mix the emotional images like image pair
        # todo cut the face and add another emotion
        # todo add random
        # todo new loss function
        # todo new optimization
        # todo new training proceudres
        # todo augment imagea fter every epoch.
        ## Freezing the first few layers. Here I am freezing the first 7 layers

        num_layers_freeze=15
        for name, child in model.named_children():
            if name=='features':
                for name, chile in child.named_children():
                    if int(name)<num_layers_freeze:
                        for params in chile.parameters():
                            params.requires_grad = False

        self.model= model
        self.learning_rate = 0.001
        self.epsilon=1
        self.optimizer = "adam"
        self.model_name_str = "inceptionv4"
        self.batch_size_train=100
        self.batch_size_test=80
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
      time.sleep(2)
      clasifier.test(epoch)
    except KeyboardInterrupt:
      clasifier.test(epoch)
      break;
    clasifier.load_data()
