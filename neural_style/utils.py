import os
import re

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
#TODO: fix
#from torch.utils.serialization import load_lua

import cv2

in_re = re.compile(r'(?:.+\.)?in\d{1,2}')

def tensor_load_rgbimage(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def tensor_load_rgbimage_cam(mImg, size=None, scale=None):
    img = mImg
    #if size is not None:
    #    img = img.resize((size, size), Image.ANTIALIAS)
    #elif scale is not None:
    #    img = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)), cv2.INTER_LINEAR)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_ret_rgbimage(tensor, cuda=False):
    img = tensor.clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    return img


def tensor_save_bgrimage(tensor, filename, cuda=False):
    tensor = torch.flip(tensor, (0,))
    tensor_save_rgbimage(tensor, filename, cuda)

def tensor_ret_bgrimage(tensor, cuda=False):
    tensor = torch.flip(tensor, (0,))
    return tensor_ret_rgbimage(tensor, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def subtract_imagenet_mean_batch(batch):
    mean = batch.new(batch.size()[1], 1, 1)
    mean[0, :, :] = 103.939
    mean[1, :, :] = 116.779
    mean[2, :, :] = 123.680
    batch = batch - mean
    return batch


def preprocess_batch(batch):
    # channels -> rgb to bgr
    return torch.flip(batch, (1,))


def init_vgg16(model_folder):
    from vgg16 import Vgg16
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system(
                'wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_folder, 'vgg16.t7'))
        raise Exception("fix lua->python loading since api was removed")
        vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src

def fix_model(state_dict):
    "remove keys put in in old pytorch versions that break 1.0"
    keys_to_delete = [key for key in state_dict if in_re.match(key) is not None]
    for key in keys_to_delete:
        del state_dict[key]
    return state_dict
