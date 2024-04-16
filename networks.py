from architectures.travnet import TravNetUp3NNRGB


def get_network(name, **kwargs):
    """ Helper function for returning network from a string """
    if name == "travnetup3nnrgb":
        return TravNetUp3NNRGB(**kwargs)
    else:
        raise ValueError("Invalid architecture")


def freeze_weights(model):
    """ Freeze weights of a model """
    for p in model.parameters():
        p.requires_grad = False
    return model
