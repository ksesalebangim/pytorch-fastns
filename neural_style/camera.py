import glob
import ast
import time
import os
import subprocess
input_path = "/home/midburn/Dropbox/midburn/camera/in/"
output_path = "/home/midburn/Dropbox/midburn/camera/out/"
old_path = "/home/midburn/Dropbox/midburn/camera/old/"
conf_path = "/home/midburn/Dropbox/midburn/camera/conf.json"
model_base_dir = "/media/midburn/benson2/pytorch/pytorch-fastns/saved-models/"
framework_dir = "/media/midburn/benson2/pytorch/pytorch-fastns/"
conf = []
f = open(conf_path,"r")
conf = f.read()
f.close()
conf = ast.literal_eval(conf)
print conf
print subprocess.Popen("mv "+output_path+"* "+old_path , shell=True, stdout=subprocess.PIPE).stdout.read()
images = (glob.glob(input_path+"*.jpg"))
for img_with_path in images:
	img = img_with_path.split("/")[-1]
	for model in conf:
		print "python3 neural_style/neural_style.py eval --content-image "+input_path+img+" --model "+model_base_dir+model+" --output-image "+output_path+img+" --cuda 1"
		subprocess.Popen("python3 neural_style/neural_style.py eval --content-image "+input_path+img+" --model "+model_base_dir+model+" --output-image "+output_path+img+" --cuda 1", shell=True, stdout=subprocess.PIPE,cwd=framework_dir).stdout.read()
print subprocess.Popen("mv "+input_path+"* "+old_path , shell=True, stdout=subprocess.PIPE).stdout.read()

