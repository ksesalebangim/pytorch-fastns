import argparse
import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import utils
from transformer_net import TransformerNet
from PIL import Image
import os
import numpy as np
import cv2
from PIL import Image
import subprocess
from effect import *
import pygame
def getRez():
    cmd = ['xrandr']
    cmd2 = ['grep', '*']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
    p.stdout.close()
     
    resolution_string, junk = p2.communicate()
    resolution = resolution_string.split()[0].decode("utf-8") 
    print(resolution)
    width, height = resolution.split('x')
    return [int(str(width)), int(str(height))]

def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)




def stylize(args):
    img = None
    postE = NoEffect()#trail(10)
    preE = resize()
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))
    cam = cv2.VideoCapture(0)


    contentScale = args.content_scale
    preEprocess =  preE.process
    utilsTensor_load_rgbimage_cam = utils.tensor_load_rgbimage_cam
    utils_preprocess_batch = utils.preprocess_batch
    utils_tensor_ret_bgrimage = utils.tensor_ret_bgrimage
    postE_process = postE.process
    cv2_resize = cv2.resize
    cv2_INTER_CUBIC = cv2.INTER_CUBIC
    args_cuda = args.cuda
    cv2_WND_PROP_FULLSCREEN = cv2.WND_PROP_FULLSCREEN
    cv2_WINDOW_FULLSCREEN = cv2.WINDOW_FULLSCREEN
    cv2_imshow = cv2.imshow
    cv2_waitKey = cv2.waitKey
    if args.cuda:
        style_model.cuda()
    rez = getRez()
    ret_val, original = cam.read()
    original=cv2.flip(original,1)
    height, width, channels = original.shape
    cv2.namedWindow("frame", cv2_WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame", cv2_WND_PROP_FULLSCREEN, cv2_WINDOW_FULLSCREEN)
    while True:
        ret_val, original = cam.read()
        original=cv2.flip(original,1)
        #original = cv2.resize(original, (int(rez[0]*0.7), int(rez[1]*0.7)), interpolation=cv2.INTER_CUBIC)
        original = preEprocess(original)
        content_image = utilsTensor_load_rgbimage_cam(original, scale=contentScale)
        content_image = content_image.unsqueeze(0)
        #if args.cuda:
        content_image = content_image.cuda()
        content_image2 = Variable(utils_preprocess_batch(content_image), volatile=True)

        output = style_model(content_image2)

        postNN = utils_tensor_ret_bgrimage(output.data[0], args_cuda)
        res = postE_process(original,postNN)
        res = cv2_resize(res, (rez[0], rez[1]), interpolation=cv2_INTER_CUBIC)

        #try to move this up
        

        cv2_imshow('frame',res)
        if cv2_waitKey(1) & 0xFF == ord('q'):
            break
        if cv2_waitKey(1) & 0xFF == ord('w'):
            preE.addSize(0.5)
        if cv2_waitKey(1) & 0xFF == ord('s'):
            preE.addSize(-0.5)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for webcam")
    main_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    main_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    main_arg_parser.add_argument("--cuda", type=int, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    main_arg_parser.add_argument("--cudnn-benchmark", action="store_true", help="use cudnn benchmark mode")

    args = main_arg_parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.cuda and args.cudnn_benchmark:
        print("Enabling cudnn benchmark mode")
        torch.backends.cudnn.benchmark = True

    try:
        stylize(args)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
