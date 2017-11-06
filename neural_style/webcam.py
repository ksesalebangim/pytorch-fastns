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
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))
    cam = cv2.VideoCapture(0)
    if args.cuda:
        style_model.cuda()

    while True:
        ret_val, img13 = cam.read()
        content_image = utils.tensor_load_rgbimage_cam(img13, scale=args.content_scale)
        content_image = content_image.unsqueeze(0)
        if args.cuda:
            content_image = content_image.cuda()
        content_image2 = Variable(utils.preprocess_batch(content_image), volatile=True)

        output = style_model(content_image2)

        im = utils.tensor_ret_bgrimage(output.data[0], args.cuda)
        cv2.imshow('frame',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


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
