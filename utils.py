from os.path import join

from torch import save, load
from torchvision import models


def save_weight(model, epoch, save_dir, file):
    save({'state_dict': model.state_dict(),
                'epoch': epoch},
                join(save_dir, file))
    

def load_weight(model, file, show = True):
    checkpoints = load(file)
    if show: print("Model at epoch:", checkpoints["epoch"])
    model.load_state_dict(checkpoints["state_dict"])
    return model


def get_pretrained(name):
    attrs = dir(models)
    check = lambda x : name + "_weights" in x.lower()
    # a = list(filter(check, attrs))

    weight_class = [attr for attr in attrs if check(attr)][0]
    weight = getattr(models, weight_class).IMAGENET1K_V1
    return weight


def get_model(name, pretrained):
    model = getattr(models, name)(weights = get_pretrained(name) if pretrained else None)
    return model