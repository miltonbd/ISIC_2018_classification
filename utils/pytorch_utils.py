import numpy as np
import torch

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


def get_random_sampler_for_classes(train_Data, num_classes):
    class_sample_count = np.repeat(0,num_classes)  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    for train_data_row in train_Data:
        index = int(train_data_row[1])
        class_sample_count[index]=class_sample_count[index]+1
    class_sample_count=class_sample_count/len(train_Data)
    class_sample_count=1/class_sample_count

    weights = []
    for train_Data_row in train_Data:
        weight=class_sample_count[int(train_data_row[1])]
        weights.append(weight)
    weights=torch.Tensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler