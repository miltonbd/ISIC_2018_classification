
import classifier
from torchsummary import summary
from torch import nn
from pretrainedmodels.models.inceptionv4 import  inceptionv4
from models.vgg import vgg19_bn
import time
from torchvision import transforms
# from data_reader_cifar import *
from classifier import Classifier
from data_reader_emotiw import *
from torch import optim
from focal_loss import FocalLoss
from augment_data import augment_images
from data_reader_isic import *
from pretrainedmodels.models.pnasnet import pnasnet5large
"""
sudo nvidia-smi -pl 180
"""
gamma = 2

def get_loss_function():
    print("==> Using Focal Loss.....")
    return FocalLoss(gamma)

def get_vgg_model():
    model=vgg19_bn(True)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    num_layers_freeze = 200

    for i,param in enumerate(model.parameters()):
        if i>num_layers_freeze:
            param.requires_grad = True
        else:
            param.requires_grad = False

    summary(model.cuda(), (3, 224, 224))
    return model

def get_pnas_large_model(gpu):
    print("==>Loading pnaslarge model...")
    model=pnasnet5large(num_classes=7)
    num_layers_freeze = 400
    for i,param in enumerate(model.parameters()):
        if i>num_layers_freeze:
            param.requires_grad = True
        else:
            param.requires_grad = False
    summary(model.cuda(), (3, 224, 224))
    return model,"pnas_large_{}_adam".format(gpu)

def get_model(gpu):
    return get_pnas_large_model(gpu)

def get_optimizer(model_trainer):
    epsilon=1e-8
    momentum = 0.9
    weight_decay=5e-4
    # model_trainer.writer.add_scalar("leanring rate", learning_rate)
    # model_trainer.writer.add_scalar("epsilon", epsilon)
    # optimizer=optim.SGD(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
    #                      lr=0.001,momentum=momentum,weight_decay=weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
                            lr=0.01)
    return optimizer

class ModelDetails(object):
    def __init__(self,gpu):
        self.model,self.model_name_str = get_model(gpu)
        self.batch_size=30
        self.epochs = 200
        self.logs_dir  = "logs/{}/{}".format(gpu,self.model_name_str)
        self.augment_images = augment_images
        self.dataset_loader=get_data_loader(self.batch_size)
        self.criterion = get_loss_function()
        self.get_optimizer = get_optimizer
        self.dataset=data_set_name

def start_training(gpu):
    model_details=ModelDetails(gpu)
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
