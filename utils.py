import random
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import vit_s, vit_s_pretrained, resnet50, resnet50_pretrained
from dataset import factory
import torchvision.transforms as transforms

MEAN = [0.4935, 0.4897, 0.4630]
STD = [0.2484, 0.2448, 0.2668]

def torch_seed(seed):
    # setting the seed of CPU
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # setting the seed of GPU
    torch.cuda.manual_seed(seed)

    # set deterministic setting of cuDNN in CUDA     
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_model(model_name, **kwargs):
    if model_name == 'vit':
        model = vit_s.VisionTransformer()
    elif model_name == 'vit42':
        model = vit_s.VisionTransformer42()
    elif model_name == 'vit_pretrained':
        model = vit_s_pretrained.PretrainedVisionTransformer()
    elif model_name == 'resnet50':
        model = resnet50.ResNet50Modified()
    elif model_name == 'resnet50_pretrained':
        model = resnet50_pretrained.PretrainedResNet50Modified(**kwargs)
    return model

def choose_optimizer(name, model, lr, betas, weight_decay):
    if name == 'AdamW':
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=tuple(map(float, ast.literal_eval(betas))),
            weight_decay=weight_decay
        )
    return optimizer

def choose_transform(model_name, transform_name):
    if model_name == 'resnet50' or model_name == 'resnet50_pretrained':
        if transform_name == None:
            tr = [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]
            
        elif transform_name == 'standard':
            tr = [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),                
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]

        elif transform_name == 'softcrop':
            tr =[
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]

    elif model_name == 'vit' or model_name == 'vit_pretrained':
        if transform_name == 'standard':
            tr = [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]

        elif transform_name == 'standard_crop':
            tr = [
                transforms.ToPILImage(),
                transforms.Resize((300, 300)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),                
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]
        elif transform_name == 'softcrop':
            tr = [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]

    elif model_name == 'vit42':     
        if transform_name == 'standard':
            tr = [
                transforms.ToPILImage(),
                transforms.Resize((42, 42)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]
        elif transform_name == 'softcrop':
            tr = [
                transforms.ToPILImage(),
                transforms.Resize((42, 42)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]

    transform = transforms.Compose(tr)

    if transform_name == 'softcrop':
        custom_tr = factory.SoftCrop()
        return transform, custom_tr
    
    return transform

def choose_criterion(transform_name):
    if transform_name == 'softcrop':
        criterion = F.kl_div
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion