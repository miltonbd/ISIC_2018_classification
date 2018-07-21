from torchsummary import summary
from models.vgg import vgg19_bn
from torch import nn
from data_reader import *
from utils.utils_all import *
from utils.pytorch_utils import *

def get_vgg_model(gpu,percentage_freeze):
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
    num_layers_freeze = 30

    params_freezed_count=0
    params_total_count=get_total_trainable_params(model)
    # for i,param in enumerate(model.parameters()):
    #     percentage_params=params_freezed_count/params_total_count
    #     if percentage_params>percentage_freeze:
    #         param.requires_grad = True
    #     else:
    #         params_freezed_count+=np.prod(param.size())
    #         param.requires_grad = False

    summary(model.cuda(), (3, height, width))
    return model,"vgg_19_{}_adam".format(gpu)

from pretrainedmodels.models.senet import senet154

def get_senet_model(gpu,percentage_freeze):
    print("==>Loading SENet model...")
    model=senet154(num_classes=num_classes)
    # params_freezed_count=0
    # params_total_count=get_total_params(model)
    # for i,param in enumerate(model.parameters()):
    #     percentage_params=params_freezed_count/params_total_count
    #     if percentage_params>percentage_freeze:
    #         param.requires_grad = True
    #     else:
    #         params_freezed_count+=np.prod(param.size())
    #         param.requires_grad = False
    """
    children(), named_children() will get iterators which may be sequential or blocks as they are added.
    modules() will returns all the raw modules from inside the sequential. 
    """
    childrens=model.named_children()
    for name,cg in childrens:
        for param in cg.parameters():
            if name=='last_linear':
                param.requires_grad = True
            else:
                param.requires_grad=False

    print("\n SENet all weights except last layer freezed.")
    show_params_trainable(model)
    # modules=model.modules()
    # for mod in modules:
    #     pass
    return model,"senet_154_{}_SGD_lr_decay".format(gpu)

def unfreeze_all_weights(model):
    params_total_count=get_total_trainable_params(model)
    for i,param in enumerate(model.parameters()):
            param.requires_grad = True
    print("All weights Unfreezed".format(params_total_count))
    show_params_trainable(model)
    return model

def freeze_percentage_weights(model,percentage_freeze):
    params_freezed_count=0
    params_total_count=get_total_trainable_params(model)
    for i,param in enumerate(model.parameters()):
        percentage_params=params_freezed_count/params_total_count
        if percentage_params>percentage_freeze:
            param.requires_grad = True
        else:
            params_freezed_count+=np.prod(param.size())
            param.requires_grad = False

    # summary(model.cuda(), (3, height, width))
    print("{}% weight freezed".format(percentage_freeze*100))
    show_params_trainable(model)
    return model

from pretrainedmodels.models.polynet import polynet

def get_polynet_model(gpu):
    print("==>Loading PolyNet model...")
    model=polynet()
    num_layers_freeze = 50
    for i,param in enumerate(model.parameters()):
        if i>num_layers_freeze:
            param.requires_grad = True
        else:
            param.requires_grad = False
    summary(model.cuda(), (3, height, width))
    return model,"senet_152_{}_adam".format(gpu)

from pretrainedmodels.models.inceptionv4 import inceptionv4

def get_inceptionv4_model(gpu):
    print("==>Loading InceptionV4 model...")
    model=inceptionv4(num_classes=7)
    num_layers_freeze = 50
    for i,param in enumerate(model.parameters()):
        if i>num_layers_freeze:
            param.requires_grad = True
        else:
            param.requires_grad = False
    summary(model.cuda(), (3, height, width))
    return model,"inception_v4_{}_adam".format(gpu)

def get_dpn_model(gpu):
    print("==>Loading DPN model...")
    model=se_resnet152(num_classes=num_classes)
    num_layers_freeze = 50
    for i,param in enumerate(model.parameters()):
        if i>num_layers_freeze:
            param.requires_grad = True
        else:
            param.requires_grad = False
    summary(model.cuda(), (3, height, width))
    return model,"senet_152_{}_adam".format(gpu)

from pretrainedmodels.models.pnasnet import  pnasnet5large

def get_pnas_large_model(gpu,percentage_freeze):
    print("==>Loading pnaslarge model...")
    model=pnasnet5large(num_classes=num_classes)
    params_freezed_count=0
    params_total_count=get_total_trainable_params(model)
    for i,param in enumerate(model.parameters()):
        percentage_params=params_freezed_count/params_total_count
        if percentage_params>percentage_freeze:
            param.requires_grad = True
        else:
            params_freezed_count+=np.prod(param.size())
            param.requires_grad = False

    print("==>{}% of weight params are freezed.".format(100*params_freezed_count/params_total_count))
    return model,"pnas_large_{}_adam".format(gpu)

from models.densenet import DenseNet201

def get_densenet_model(gpu):
    print("==>Loading DenseNet@01 model...")
    model=DenseNet201(num_classes=num_classes)
    num_layers_freeze = 400
    for i,param in enumerate(model.parameters()):
        if i>num_layers_freeze:
            param.requires_grad = True
        else:
            param.requires_grad = False
    summary(model.cuda(), (3, height, width))
    return model,"densenet_{}_adam".format(gpu)