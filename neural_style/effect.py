import cv2
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import operator
from PIL import Image
from PIL import ImageDraw
import numpy as np
import pygame
import subprocess

class Pre_Effect(object):
	def init():
		pass

	def process(self,preEffect):
		pass

class Post_Effect(object):
	def init():
		pass

	def process(self,preEffect,post_Effect):
		pass

class trail(Post_Effect):
	def __init__(self,stackSize=10):
		self.stackSize = stackSize
		self.stack = []
		self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

	def setStackSize(self,stackSize):
		self.stackSize = stackSize

	def process(self,preEffect,post_Effect):
		mask = self.fgbg.apply(preEffect)
		self.stack.append(mask)
		while len(self.stack) > self.stackSize:
			del self.stack[0]
		for x in range(len(self.stack)):
			mask = cv2.bitwise_or(mask, self.stack[x])
		fg = cv2.bitwise_or(post_Effect, post_Effect, mask=mask)
		mask = cv2.bitwise_not(mask)
		bk = cv2.bitwise_or(preEffect, preEffect, mask=mask)
		final = cv2.bitwise_or(fg, bk)
		return final

class paintMovement(Post_Effect):
	def __init__(self):
		self.stack = []
		self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
		self.bgEfect = None
		self.bgOriginal = None



	def process(self,preEffect,post_Effect):
		if self.bgOriginal is None:
			self.bgEfect = post_Effect
			self.bgOriginal = cv2.cvtColor(preEffect, cv2.COLOR_BGR2GRAY)
		grayB = cv2.cvtColor(preEffect, cv2.COLOR_BGR2GRAY)
		(score, diff) = compare_ssim(self.bgOriginal, grayB, full=True)
		diff = (diff * 255).astype("uint8")

		mask = cv2.threshold(diff, 0, 255,
			 cv2.THRESH_OTSU)[1]
		mask = cv2.medianBlur(mask, 3)
		fg = cv2.bitwise_or(post_Effect, post_Effect, mask=mask)
		mask = cv2.bitwise_not(mask)
		bk = cv2.bitwise_or(self.bgEfect, self.bgEfect, mask=mask)
		#bk = self.bgEfect
		final = cv2.bitwise_or(fg, bk)
		return final
		#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		#	cv2.CHAIN_APPROX_SIMPLE)
		#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
class NoEffect(Post_Effect):
	def __init__(self):
		pass

	def process(self,preEffect,post_Effect):
		return post_Effect

		
class noPre(Pre_Effect):
	def __init__(self):
		pass
	def process(self,preEffect):
		return preEffect

class tripleEffect(Pre_Effect):
	def __init__(self):
		self.maxSize = 200
		self.currentSize = 0
		self.direction = 1
		self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
		pass
	def process(self,preEffect):
		mask = self.fgbg.apply(preEffect)
		if self.currentSize >= (self.maxSize/2) and self.direction==1:
			self.direction=0
		if self.currentSize == 0 and self.direction==0:
			self.direction=1
		if self.direction==1:
			self.currentSize+=2
		else:
			self.currentSize-=2

		t = preEffect.shape
		t0=t[0]+self.maxSize
		t1=t[1]+self.maxSize

		img5 = cv2.bitwise_or(preEffect, preEffect, mask=mask)

		img1 = np.zeros((t0, t1, 3))
		img2 = np.zeros((t0, t1, 3))
		img3 = np.zeros((t0, t1, 3))
		img4 = np.zeros((t0, t1, 3))

		x_offset=int(self.maxSize/2) + self.currentSize
		y_offset=self.maxSize
		img1[y_offset:y_offset+preEffect.shape[0], x_offset:x_offset+preEffect.shape[1]] = img5


		x_offset=int(self.maxSize/2)
		y_offset=self.maxSize - self.currentSize
		img2[y_offset:y_offset+preEffect.shape[0], x_offset:x_offset+preEffect.shape[1]] = img5

		x_offset=int(self.maxSize/2) - self.currentSize
		y_offset=self.maxSize
		img3[y_offset:y_offset+preEffect.shape[0], x_offset:x_offset+preEffect.shape[1]] = img5

		x_offset=int(self.maxSize/2)
		y_offset=self.maxSize
		img4[y_offset:y_offset+preEffect.shape[0], x_offset:x_offset+preEffect.shape[1]] = preEffect


		dst1 = cv2.addWeighted(img1,0.5,img2,0.5,0)
		dst2 = cv2.addWeighted(img3,0.5,img4,0.5,0)
		dst3 = cv2.addWeighted(dst2,0.5,dst1,0.5,0)
		#dst = cv2.addWeighted(preEffect,0.7,img2,0.3,0)
		return dst3


class preserve_color(Post_Effect):
	def __init__(self):
		pass

	def process(self,preEffect,post_Effect):
		content_yuv = cv2.cvtColor(np.float32(preEffect), cv2.COLOR_RGB2YUV)
		post_Effect = np.squeeze(post_Effect)
		post_Effect = post_Effect[:,:,(2,1,0)]  # bgr to rgb
		post_Effect = post_Effect + [103.939, 116.779, 123.68]
		if content_yuv is not None:
			yuv = cv2.cvtColor(np.float32(post_Effect), cv2.COLOR_RGB2YUV)
			yuv[:,:,1:3] = content_yuv[:,:,1:3]
			post_Effect = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
			post_Effect = np.clip(post_Effect, 0, 255).astype(np.uint8)
		return post_Effect

class resize(Pre_Effect):
	def __init__(self):
		self.imsize = 0.7
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
		t=getRez()
		self.width=t[0]
		self.height=t[1]

	def process(self,preEffect):
		if self.imsize == 1:
			return preEffect
		height, width, channels = preEffect.shape
		return cv2.resize(preEffect, (int(self.width*self.imsize), int(self.height*self.imsize)), interpolation=cv2.INTER_CUBIC)

	def addSize(self,size):
		self.imsize+=size

