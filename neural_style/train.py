from os import walk
import os
from os import listdir
from os.path import isfile, join
import subprocess
import sys
import time
import shlex, subprocess
import glob


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    trained = False
    d = "/home/midburn/Dropbox/midburn"
    inputd = d + "/input_images"
    directoriez = [os.path.join(inputd, o) for o in os.listdir(inputd) if os.path.isdir(os.path.join(inputd, o))]
    for direc in directoriez:
        onlysubdir = direc.split("/")[-1]
        if len(onlysubdir.split("_")) >= 3:
            onlyfiles = [f for f in listdir(direc) if isfile(join(direc, f))]
            for myfile in onlyfiles:
                trained = True
                if myfile[-4:].lower() == ".jpg":
                    os.rename(inputd + "/" + onlysubdir + "/" + myfile,
                              inputd + "/" + onlysubdir + "/" + myfile + ".trained")
                    arguments = onlysubdir.split("_")

                    command = "python3 neural_style/neural_style.py train --style-image " + inputd + "/" + onlysubdir + "/" + myfile + ".trained --dataset /media/midburn/benson2/fast-neural-style/coco/ --content-weight " + \
                                                      arguments[1] + " --style-weight " + arguments[2] + " --batch-size 1 --epochs " + \
                                                      arguments[0] + " --save-model-dir saved-models/" + \
                                                    " --cuda 1 --cudnn-benchmark --log-interval 100 --image-size 512 --vgg-model-dir /media/midburn/benson2/pytorch/pytorch-fastns/vgg/"
                    print(command)
                    command = shlex.split(command)
                    sub_process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                                   cwd="/media/midburn/benson2/pytorch/pytorch-fastns")
                    modelFileName = ""
                    while sub_process.poll() is None:
                        out = sub_process.stdout.readline()
                        out = out.decode("utf-8") 
                        if "trained model saved at " in out:
                            modelFileName = (out.split("trained model saved at ")[1]).strip()
                            if "saved-models/" in modelFileName:
                                modelFileName = modelFileName.split("saved-models/")[1]
                        sys.stdout.flush()
                        print(out)
                    
                    outd = d + "/output_images/" + onlysubdir
                    ensure_dir(outd + "/")

                    
                    command = "python3 neural_style/neural_style.py eval --content-image /media/midburn/benson2/pytorch/pytorch-fastns/baseline.jpg"+\
                                " --model saved-models/"+modelFileName+" --output-image "+outd+"/"+modelFileName+".jpg --cuda 1" 
                    print(command)
                    command = shlex.split(command)
                    sub_process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                                   cwd="/media/midburn/benson2/pytorch/pytorch-fastns")
                    while sub_process.poll() is None:
                        out = sub_process.stdout.readline()
                        out = out.decode("utf-8") 
                        print(out)


                   
                  
                   
                    #f = []
                    """for (dirpath, dirnames, filenames) in walk("/media/midburn/fast-neural-style/mymodels/"+onlysubdir):
                        f.extend(filenames)
			break
                    for myfile1 in f:
                        # print myfile1
                        finder = str(myfile).split(".jpg")[0].split("/")[-1]
                        #print finder
                        if str(myfile1).startswith(finder) and str(myfile1).endswith(".t7"):
                            interation = str(myfile1).split("_")[-1].split(".t7")[1]
                            command = "th fast_neural_style.lua -model " + "mymodels/" + onlysubdir + "/" + myfile1  + " -input_image baseline.jpg -output_image " + outd + "/" + myfile1[:-3]+".jpg" + " -gpu 0"
                            print command
                            command = command.split(" ")
                            sub_process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                                           cwd="/media/midburn/fast-neural-style")
                            while sub_process.poll() is None:
                                out = sub_process.stdout.read(1)
                                sys.stdout.write(out)
                                sys.stdout.flush()"""
                    print("trained " + "train_script/" + onlysubdir + "/" + myfile)
                    time.sleep(2)  # Sleep 5 minutes to let the pc cool after trainning a model
                    return trained
def continue_train():
    path_to_models = "/media/midburn/benson2/pytorch/pytorch-fastns/saved-models"
    current = "/current"
    old = "/old"
    files = glob.glob(path_to_models+current+"/*.model*")
    it = 0
    currModel = ""
    for x in files:
        iteration = x.split("@@@@@")[1]
        if int(iteration) > it:
            it = iteration
            currModel = x
    try:
        image = currModel.split(".model")[0]
        filename = currModel.split("/")[-1]
        splitted = currModel.split("/")[-1].split("_")

        style = splitted[1]
        content  = splitted[0]
        command = "mv "+x+" "+path_to_models+current+"/"
        command = shlex.split(command)
        output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]

        command = "python3 neural_style/neural_style.py train --style-image " + image +" --dataset /media/midburn/benson2/fast-neural-style/coco/ --content-weight " + \
                                                          content + " --style-weight " + style + " --batch-size 1 --epochs 1 --cuda 1 --cudnn-benchmark --log-interval 100 --image-size 512 --vgg-model-dir /media/midburn/benson2/pytorch/pytorch-fastns/vgg/" + \
                                                          " --model "+current+old+"/"+filename
        print(command)
        command = shlex.split(command)
        sub_process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       cwd="/media/midburn/benson2/pytorch/pytorch-fastns")
        modelFileName = ""
        while sub_process.poll() is None:
            out = sub_process.stdout.readline()
            out = out.decode("utf-8") 
            if "trained model saved at " in out:
                modelFileName = (out.split("trained model saved at ")[1]).strip()
                if "saved-models/" in modelFileName:
                    modelFileName = modelFileName.split("saved-models/")[1]
            sys.stdout.flush()
            print(out)
    except Exception as e:
        pass
command = "ps aux"
command = shlex.split(command)
sub_process = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
#if "neural_style.py" in sub_process:
#    exit(0)
while True:
    hasTrained = main()
    if hasTrained == True:
        time.sleep(2)
    else:
        continue_train()


#python3 neural_style/neural_style.py train --dataset /media/midburn/benson2/fast-neural-style/coco_part/ --vgg-model-dir vgg --batch-size 4 --save-model-dir saved-models --cuda 1 --log-interval 50 --cudnn-benchmark
