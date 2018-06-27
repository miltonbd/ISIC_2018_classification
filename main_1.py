import  os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import classifier

from torch import nn
from pretrainedmodels.models.inceptionv4 import  inceptionv4
from models.vgg import vgg19_bn
from augment_data_isic import augment_images
import time
from torchvision import transforms
from data_reader_cifar import *
from classifier import Classifier
from torch import optim

"""
sudo nvidia-smi -pl 180
"""

def get_inception_v4_model():
    model = inceptionv4()
    model.avg_pool = nn.AvgPool2d(2, count_include_pad=False)
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

    num_layers_freeze = 15
    for name, child in model.named_children():
        if name == 'features':
            for name, chile in child.named_children():
                if int(name) < num_layers_freeze:
                    for params in chile.parameters():
                        params.requires_grad = False
    return model

def get_vgg_model():
    model=vgg19_bn(True)
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model

def get_model():
    return get_vgg_model()

def get_optimizer(classifier):
    learning_rate=0.001
    epsilon=1e-8
    classifier.writer.add_scalar("leanring rate", learning_rate)
    classifier.writer.add_scalar("epsilon", epsilon)
    return  optim.SGD(classifier.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

class ModelDetailsInceptionV4(object):
    def __init__(self):

        self.model= get_model()
        self.model_name_str = "inceptionv4"
        self.batch_size=100
        self.epochs = 200
        self.logs_dir  = "logs/inceptionv4/no_aug"
        self.augmentor = augment_images
        self.dataset_loader=get_data_loader(self.batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.get_optimizer = get_optimizer

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
