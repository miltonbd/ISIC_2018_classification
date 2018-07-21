import numpy as np

def get_total_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return  params

def get_total_params(model):
    model_parameters = model.parameters()
    params = sum([np.prod(p.size()) for p in model_parameters])
    return  params

def show_params_trainable(model):
    total=get_total_params(model)
    trainbale=get_total_trainable_params(model)
    print("\n Total Params:        {}".format(total))
    print("\n Trainable Params:    {}".format(trainbale))
    print("\n Non Trainable Params:{}".format(total-trainbale))