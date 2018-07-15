

def load_pretrained_dict_only_matched(model, pretrained_dict):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    last_layer_keys=pretrained_dict.keys()
    pretrained_dict.pop("last_linear.bias")
    pretrained_dict.pop("last_linear.weight")
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model