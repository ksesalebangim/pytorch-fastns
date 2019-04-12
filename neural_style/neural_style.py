import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from networks import ResNeXtNet
from vgg16 import Vgg16
import mysql.connector
import os.path
from utils import fix_model

import sys

def getEpoch(filename):
    return int(filename.split("/")[-1].split("@@@@@@")[1])

class VisdomLinePlotter:
    """Plots to Visdom"""
    def __init__(self, plot_name, y_axis='loss', env_name='main'):
        import visdom
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plot_name = plot_name
        self.y_axis = y_axis
        self.window = None

    def plot(self, var_name, split_name, x, y):
        name = "{} {}".format(var_name, split_name)
        if self.window is None:
            self.window = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[name],
                title=self.plot_name,
                xlabel='Epochs',
                ylabel=self.y_axis
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.window, name=name, update='append')

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    serialNumFile = "serialNum.txt"
    serial = 0
    if os.path.isfile(serialNumFile):
        with open(serialNumFile,"r") as t:
            serial = int(t.read())

    serial +=  1
    with open(serialNumFile,"w") as t:
        t.write(str(serial))
        
    if args.mysql:
        cnx = mysql.connector.connect(user='root', database='midburn',password='midburn')
        cursor = cnx.cursor()
    location = args.dataset.split("/")
    if location[-1]=="":
        location=location[-2]
    else:
        location=location[-1]
    save_model_filename = str(serial)+"_"+extractName(args.style_image)+"_"+str(args.epochs) + "_" + str(int(
        args.content_weight)) + "_" + str(int(args.style_weight)) +"_size_"+str(args.image_size)+"_dataset_"+str(location)+ ".model"
    print(save_model_filename)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    m_epoch = 0
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        #kwargs = {'num_workers': 0, 'pin_memory': False}
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    transformer = TransformerNet()
    #transformer = ResNeXtNet()
    transformer_type = transformer.__class__.__name__
    optimizer = Adam(transformer.parameters(), args.lr)
    if args.l1:
        loss_criterion = torch.nn.L1Loss()
    else:
        loss_criterion = torch.nn.MSELoss()
    loss_type = loss_criterion.__class__.__name__

    if args.visdom:
        vis = VisdomLinePlotter("Style Transfer: " + transformer_type)
    else:
        vis = None

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    if args.model is not None:
        transformer.load_state_dict(torch.load(args.model))
        save_model_filename = save_model_filename + "@@@@@@"+str(int(getEpoch(args.model))+int(args.epochs))
        m_epoch+=int(getEpoch(args.model))
        print("loaded model\n")

    for param in vgg.parameters():
        param.requires_grad = False

    with torch.no_grad():
        style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
        style = style.repeat(args.batch_size, 1, 1, 1)
        style = utils.preprocess_batch(style)
        if args.cuda:
            style = style.cuda()
        
        style = utils.subtract_imagenet_mean_batch(style)
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]
        del features_style
        del style

    # TODO: scheduler and style-loss criterion unused at the moment
    scheduler = StepLR(optimizer, step_size=15000//args.batch_size)
    style_loss_criterion = torch.nn.CosineSimilarity()
    total_count = 0

    if args.mysql:
        q1 = ("REPLACE INTO `images`(`name`) VALUES ('"+args.style_image+"')")
        cursor.execute(q1)
        cnx.commit()
        imgId = cursor.lastrowid

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):

            n_batch = len(x)
            count += n_batch
            total_count += n_batch
            optimizer.zero_grad()
            x = utils.preprocess_batch(x)
            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(x)

            features_y = vgg(y)
            f_xc_c = vgg.content_features(xc)

            content_loss = args.content_weight * loss_criterion(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = gram_style[m]
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += loss_criterion(gram_y, gram_s[:n_batch, :, :])
                #style_loss -= style_loss_criterion(gram_y, gram_s[:n_batch, :, :])

            style_loss *= args.style_weight
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            # TODO: enable
            #scheduler.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                if args.mysql:
                    q1 = ("REPLACE INTO `statistics`(`imgId`,`epoch`, `iteration_id`, `content_loss`, `style_loss`, `loss`) VALUES ("+str(imgId)+","+str(int(e)+m_epoch)+","+str(batch_id)+","+str(agg_content_loss / (batch_id + 1))+","+str(agg_style_loss / (batch_id + 1))+","+str((agg_content_loss + agg_style_loss) / (batch_id + 1))+")")
                    cursor.execute(q1)
                    cnx.commit()
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}\n".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                sys.stdout.flush()
                print(mesg)
            if vis is not None:
                vis.plot(loss_type, "Content Loss", total_count, content_loss.item())
                vis.plot(loss_type, "Style Loss", total_count, style_loss.item())
                vis.plot(loss_type, "Total Loss", total_count, total_loss.item())

    # save model
    transformer.eval()
    transformer.cpu()
    
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def extractName(nameAndPath):
    splitted = nameAndPath.split("/")
    if len(splitted) == 1:
        return splitted[0]
    print(splitted[-1])
    return splitted[-1]

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
    content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.content_scale)
    content_image = content_image.unsqueeze(0)

    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
    style_model = TransformerNet()
    style_model.load_state_dict(fix_model(torch.load(args.model)))

    if args.cuda:
        style_model.cuda()

    output = style_model(content_image)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
                                  help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--style-weight", type=float, default=5.0,
                                  help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--cudnn-benchmark", action="store_true", help="use cudnn benchmark mode")
    train_arg_parser.add_argument("--visdom", action="store_true", help="plot training losses to visdom")
    train_arg_parser.add_argument("--l1", action="store_true", help="Use L1 instead of mse loss")
    train_arg_parser.add_argument("--model", type=str, default=None,
                                  help="continue training a model")
    train_arg_parser.add_argument("--no-mysql", action="store_false", dest="mysql",
                                  help="use mysql to store statistics")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--cudnn-benchmark", action="store_true", help="use cudnn benchmark mode")


    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.cuda and args.cudnn_benchmark:
        print("Enabling cudnn benchmark mode")
        torch.backends.cudnn.benchmark = True

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
#python neural_style/neural_style.py eval --content-image /media/midburn/benson2/pytorch/pytorch-fastns/baseline.jpg --model saved-models/pink.model --output-image /home/midburn/test.jpg --cuda 1
