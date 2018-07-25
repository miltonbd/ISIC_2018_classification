import  os
gpu=1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from classifier import Classifier
from torch import optim
from model_loader import *
from loss_loader import *
from torch.optim import lr_scheduler

"""
sudo nvidia-smi -pl 220
use command line to run the training.

################## ToDo ########################
    1. download more images using image_utils and isic-arhive. Also, use more online resources for data. 
    2. use additional dasets used in https://github.com/learningtitans/isbi2017-part3
    3. use pair augmentation, random erase (data augmentaton leads to poor accuracy if model has less capacity.)
    4. download more images for each classes.
    5. preprocessing and feature extraction
    6. bigger 500 px image size. big image tends to make
    7. save model and load from previous
    8. adversarial training, use crosssentropy, focal loss
    9. train atleast 20 epoch.
    10.BGR to RGB
    11. subtract imagenet mean and divide by sd     
"""

def get_loss_function(classifier):
    return get_cross_entropy(classifier)

def get_model(gpu):
    return get_pnas_large_model(gpu)

def get_optimizer(model_trainer):
    momentum = 0.7
    weight_decay=5e-9
    # model_trainer.writer.add_scalar("leanring rate", learning_rate)
    # model_trainer.writer.add_scalar("epsilon", epsilon)

    params=filter(lambda p: p.requires_grad, model_trainer.model.parameters())
    optimizer = optim.Adam(params,
                                lr=0.0001)
    # optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # set up scheduler

    #todo SGD
    return (optimizer,scheduler)

class ModelDetails(object):
    def __init__(self,gpu=None):
        self.gpu=gpu
        model, model_name = get_model(gpu)
        self.model=freeze_all_weighs_except_last_layer(model)
        self.model_name_str=model_name
        self.batch_size=4
        self.epochs = 500
        self.logs_dir  = "logs/{}/{}".format(gpu,self.model_name_str)
        self.dataset_loader=get_data_loader(self.batch_size)
        self.get_loss_function = get_loss_function
        self.get_optimizer = get_optimizer
        self.dataset=data_set_name
        self.weight_freeze_epochs=2

def start_training(gpu):
    model_details=ModelDetails(gpu)
    clasifier=Classifier(model_details)
    clasifier.load_data()
    clasifier.load_model()
    for epoch in range(clasifier.start_epoch, clasifier.start_epoch + model_details.epochs):
        if epoch < model_details.weight_freeze_epochs:
            freeze_all_weighs_except_last_layer(model_details.model)
        else:
            unfreeze_all_weights(model_details.model)

        # if epoch >1:
        #     freeze_percentage_weights(clasifier.model, 0.3)
        try:
          clasifier.train(epoch)
          # clasifier.validate(epoch)
          clasifier.test(epoch)
        except KeyboardInterrupt:
          clasifier.test(epoch)
          break;
        clasifier.load_data()

start_training(gpu)
