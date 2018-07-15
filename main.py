import  os
gpu=0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from classifier import Classifier
from torch import optim
from augment_data import augment_images
from model_loader import *
from loss_loader import *
from torch.optim import lr_scheduler

"""
sudo nvidia-smi -pl 180
use command line to run the training.

################## ToDo ########################
    1. download more images using image_utils and isic-arhive. Also, use more online resources for data. 
    2. use additional dasets used in https://github.com/learningtitans/isbi2017-part3
    3. use pair augmentation, random erase
    4. download more images for each classes.
    5. preprocessing and feature extraction
    6. bigger 500 px image size. big image tends to make
    7. save model and load from previous
    8. adversarial training, use crosssentropy, focal loss
    9. use similar optimizatio adam and learning rate schedule like wider face pedestrian dataset. (done)
    10.BGRto RGB
    11.     
    

"""

def get_loss_function(classifier):
    return get_cross_entropy(classifier)

def get_model(gpu):
    return get_pnas_large_model(gpu,.3)

def get_optimizer(model_trainer):
    epsilon=1e-8
    momentum = 0.9
    weight_decay=5e-4
    # model_trainer.writer.add_scalar("leanring rate", learning_rate)
    # model_trainer.writer.add_scalar("epsilon", epsilon)
    # optimizer=optim.SGD(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
    #                      lr=0.001,momentum=momentum,weight_decay=weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
                            lr=0.001)
    return optimizer

class ModelDetails(object):
    def __init__(self,gpu):
        self.model,self.model_name_str = get_model(gpu)
        self.batch_size=20
        self.epochs = 200
        self.logs_dir  = "logs/{}/{}".format(gpu,self.model_name_str)
        self.augment_images = augment_images
        self.dataset_loader=get_data_loader(self.batch_size)
        self.get_loss_function = get_loss_function
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

start_training(1)
