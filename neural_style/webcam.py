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
from networks import ResNeXtNet
from PIL import Image
import os
import numpy as np
import cv2
from PIL import Image
import subprocess
from effect import *
import pygame
import os
import time
from mytimer import mytimer
from utils import fix_model
import json
from time import sleep

myTimer = None
cv2_WND_PROP_FULLSCREEN = cv2.WND_PROP_FULLSCREEN
cv2_WINDOW_FULLSCREEN = cv2.WINDOW_FULLSCREEN
cv2.namedWindow("frame", cv2_WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2_WND_PROP_FULLSCREEN, cv2_WINDOW_FULLSCREEN)
mTimer = mytimer()
currrDir=(os.path.dirname(os.path.realpath(__file__)))
print(currrDir+"/screensaver/1.jpg")
global screensaverImg
screensaverImg = cv2.imread(currrDir+"/screensaver/1.jpg")
def getRez():
    cmd = ['xrandr']
    cmd2 = ['grep', '*']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
    p.stdout.close()
     
    resolution_string, junk = p2.communicate()
    resolution = resolution_string.split()[0].decode("utf-8") 
    #print(resolution)
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



def stylize(modelsJsonPath, model=None):
    # TODO: hack for local testing
    if model is not None:
        mTimer.setBuffer(100000)
        stylizeModel(model, None, None)
        return

    f = open(modelsJsonPath,"r")
    data = f.read()
    f.close()
    asJson = json.loads(data)
    while True:
        for element in asJson:
            #print(element["model"])
            scale = 1
            sleep_time = 0.0
            timeDelay = 30
            if "scale" in element:
                scale = element["scale"]
            if "buffer" in element:
                mTimer.setBuffer(int(element["buffer"]))
            else:
                mTimer.setBuffer(60*5) # If no buffer is set then default is 5 minutes
            # Some models needs to step down in speed. Others dont
            if "sleep" in element:
               sleep_time = element["sleep"]

            if(stylizeModel(element["model"],float(scale), float(sleep_time))):
                return


def stylizeModel(model,content_scale=None, sleep_time=0):
    img = None
    postE = NoEffect()#trail(10)
    preE = resize()
    style_model = TransformerNet()
    style_model.load_state_dict(fix_model(torch.load(model)))
    cam = cv2.VideoCapture(0)

    preEprocess =  preE.process
    utilsTensor_load_rgbimage_cam = utils.tensor_load_rgbimage_cam
    utils_preprocess_batch = utils.preprocess_batch
    utils_tensor_ret_bgrimage = utils.tensor_ret_bgrimage
    postE_process = postE.process
    cv2_resize = cv2.resize
    cv2_INTER_CUBIC = cv2.INTER_CUBIC
    #args_cuda = args.cuda

    cv2_imshow = cv2.imshow
    cv2_waitKey = cv2.waitKey
    #if args.cuda:
    style_model.cuda()
    rez = getRez()
    width1 = rez[0]
    height1 = rez[1]
    mysize = (width1, height1)
    ret_val, original = cam.read()
    original=cv2.flip(original,1)
    height, width, channels = original.shape
    delayTest = 0
    #original=cv2.flip(original,1)
    #original = preEprocess(original)

    with torch.no_grad():
        # trace once for great justice
        ret_val, original = cam.read()

        content_image = utils.tensor_load_rgbimage_cam(original, scale=content_scale)
        content_image = content_image.unsqueeze(0)
        content_image = content_image.cuda()
        content_image = utils.preprocess_batch(content_image)

        traced_net = torch.jit.trace(style_model,(content_image,))

        # now loop
        while True:
            if(sleep_time):
                sleep(sleep_time)

            ret_val, original = cam.read()
            # TODO: hack
            #original = original[140:340, 160:480]
            #original = cv2.resize(original, (480,640), cv2.INTER_LINEAR)
            #original = cv2.resize(original, (1080,1920), cv2.INTER_LINEAR)
            # TODO: re-enable? make parameter?
            #original = cv2.flip(original, 1)
            #original = preEprocess(original)

            content_image = utils.tensor_load_rgbimage_cam(original, scale=content_scale)
            content_image = content_image.unsqueeze(0)
            content_image = content_image.cuda()
            content_image = utils.preprocess_batch(content_image)

            #output = style_model(content_image2)
            output = traced_net(content_image)

            res = utils.tensor_ret_bgrimage(output[0])
            #res = postE_process(original,postNN)

            res = cv2_resize(res, mysize , interpolation=cv2_INTER_CUBIC)

            #try to move this up
    
            cv2_imshow('frame',res)
            pressed_key = cv2_waitKey(1) & 0xFF;
            if pressed_key == ord('q'):
                return True
            # On space move to next model
            elif pressed_key == 32:
                return False

            delayTest += 1
            if delayTest % 30 == 0:

                delayTest = 0
                if mTimer.isTransition():
                    #cv2.waitKey(1)
                    return

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for webcam")
    main_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    main_arg_parser.add_argument("--model", type=str, required=False,
                                 help="saved model to be used for stylizing the image. overrides config file")
    main_arg_parser.add_argument("--config-path", type=str, default="/media/midburn/benson2/pytorch/pytorch-fastns/conf/1.json",
                                 help="path to config json")
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
        #stylize(args)
        stylize(args.config_path, args.model)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
